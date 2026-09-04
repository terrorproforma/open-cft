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

from collections import deque
from dataclasses import dataclass
from math import gcd, isfinite, pi, sqrt
from typing import Any, Literal, Mapping

import numpy as np

from . import kernels
from .fields import MagneticFieldMap
from .mcc import MCCConfig, NullCollisionMCC, XenonCrossSections, maxwellian_velocity
from .mesh import MeshMasks, build_mesh_masks, cell_index
from .neutrals import NeutralInventory, NeutralInventoryConfig, NeutralState
from .models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    EV_J,
    BoundaryPotentials,
    Grid2D,
    PIC2DStabilityError,
    PIC2DValidationError,
    ParticleArrays,
    PoissonConfig2D,
    Species2D,
    StabilityLimits,
    electron_species,
    require_stable,
    stability_report,
    xenon_ion_species,
)
from .poisson import Poisson2D, apply_operator, electric_field_nodes, field_energy_j, induced_electrode_charge_c
from .sensitivity import AnomalousCollisionConfig, SEEConfig, apply_bohm_scattering

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
class CathodeConfig:
    """v2.0 cathode / neutraliser: an electron emission region inside the plume.

    Electrons are born uniformly in the volume of the annulus
    ``r_inner_m <= r <= r_outer_m, z_start_m <= z <= z_end_m`` with an isotropic
    Maxwellian at ``electron_temperature_ev`` (HEMP-T neutralisers sit off-axis
    outside the exit; Kornfeld, Koch and Harmann 2007).  ``current_rule``:

    * ``fixed``: ``current_a`` at every step (the v1.x rule moved into the plume);
    * ``continuity``: the emitted current follows the discharge (anode) current of
      the previous series interval, so that in steady state the plume boundary
      carries no net current to the chamber (the neutraliser is in series with the
      anode supply; Szabo 2001; Charoy et al. 2019; literature review blocker 4d
      variant (c)).  The rate is relaxed with an exponential moving average over
      ``continuity_relaxation_intervals`` series intervals and clamped to
      ``[current_a (floor, also the ignition current), max_current_a]``.
    """

    r_inner_m: float
    r_outer_m: float
    z_start_m: float
    z_end_m: float
    electron_temperature_ev: float
    current_a: float
    current_rule: Literal["fixed", "continuity"] = "fixed"
    max_current_a: float | None = None
    continuity_relaxation_intervals: float = 4.0

    def __post_init__(self) -> None:
        for name in ("r_inner_m", "r_outer_m", "z_start_m", "z_end_m", "electron_temperature_ev", "current_a", "continuity_relaxation_intervals"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise PIC2DValidationError(f"cathode {name} must be a finite number")
        if not 0.0 <= self.r_inner_m < self.r_outer_m:
            raise PIC2DValidationError("cathode annulus needs 0 <= r_inner_m < r_outer_m")
        if not self.z_start_m < self.z_end_m:
            raise PIC2DValidationError("cathode annulus needs z_start_m < z_end_m")
        if self.electron_temperature_ev <= 0.0:
            raise PIC2DValidationError("cathode electron temperature must be positive")
        if self.current_a < 0.0:
            raise PIC2DValidationError("cathode current must be non-negative")
        if self.current_rule not in ("fixed", "continuity"):
            raise PIC2DValidationError("cathode current_rule must be 'fixed' or 'continuity'")
        if self.current_rule == "continuity":
            if self.max_current_a is None or not isfinite(self.max_current_a) or self.max_current_a < self.current_a:
                raise PIC2DValidationError("continuity emission needs max_current_a >= current_a (floor)")
            if self.continuity_relaxation_intervals < 1.0:
                raise PIC2DValidationError("continuity_relaxation_intervals must be >= 1")
        elif self.max_current_a is not None:
            raise PIC2DValidationError("max_current_a applies to the continuity rule only")

    @property
    def peak_current_a(self) -> float:
        return self.current_a if self.max_current_a is None else float(self.max_current_a)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "r_inner_m": self.r_inner_m, "r_outer_m": self.r_outer_m, "z_start_m": self.z_start_m, "z_end_m": self.z_end_m,
            "electron_temperature_ev": self.electron_temperature_ev, "current_a": self.current_a, "current_rule": self.current_rule,
        }
        if self.current_rule == "continuity":
            record["max_current_a"] = float(self.max_current_a)
            record["continuity_relaxation_intervals"] = self.continuity_relaxation_intervals
        return record


@dataclass(frozen=True, slots=True)
class SeedPlasmaConfig:
    density_per_m3: float
    electron_temperature_ev: float
    ion_temperature_ev: float = 0.0
    # v2.0: "channel" seeds the channel volume only (z < L_channel); the plume starts empty and fills from
    # the beam and the cathode.  "all" (default, the v1.x identity) seeds the whole plasma region.
    region: Literal["all", "channel"] = "all"

    def __post_init__(self) -> None:
        if not isfinite(self.density_per_m3) or self.density_per_m3 < 0.0:
            raise PIC2DValidationError("seed density must be finite and non-negative")
        if not isfinite(self.electron_temperature_ev) or self.electron_temperature_ev <= 0.0:
            raise PIC2DValidationError("seed electron temperature must be positive")
        if not isfinite(self.ion_temperature_ev) or self.ion_temperature_ev < 0.0:
            raise PIC2DValidationError("seed ion temperature must be non-negative")
        if self.region not in ("all", "channel"):
            raise PIC2DValidationError("seed region must be 'all' or 'channel'")

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "density_per_m3": self.density_per_m3,
            "electron_temperature_ev": self.electron_temperature_ev,
            "ion_temperature_ev": self.ion_temperature_ev,
        }
        if self.region != "all":   # v1.x config identity unchanged for the default
            record["region"] = self.region
        return record


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
    # v1.4: fail-closed runtime gate on cells per Debye length at the peak-density node
    # (evaluated at every series record); None = recorded only when a gate is absent.
    peak_debye_gate: "PeakDebyeGateConfig | None" = None
    # v1.4 sensitivity hooks (default OFF): Bohm-type anomalous scattering and the SEE scaffold.
    anomalous: "AnomalousCollisionConfig | None" = None
    see: "SEEConfig | None" = None
    # v2.0: cathode emission region in the plume (replaces the exit-plane ``injection``, which stays
    # as the legacy A/B option); requires a plume geometry.
    cathode: "CathodeConfig | None" = None
    # v2.0: fail-closed charge pile-up gate on the far-field boundary (plume geometries)
    plume_boundary_gate: "PlumeBoundaryGateConfig | None" = None

    def __post_init__(self) -> None:
        if not isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise PIC2DValidationError("dt_s must be positive")
        if self.plume_boundary_gate is not None and not self.grid.geometry.has_plume:
            raise PIC2DValidationError("the plume boundary gate requires a plume geometry")
        if self.cathode is not None:
            if self.injection is not None:
                raise PIC2DValidationError("use either the v2.0 cathode emission region or the legacy exit-plane injection, not both")
            geometry = self.grid.geometry
            if not geometry.has_plume:
                raise PIC2DValidationError("the cathode emission region requires a plume geometry")
            if not (geometry.z_max_m <= self.cathode.z_start_m and self.cathode.z_end_m <= geometry.domain_z_max_m
                    and self.cathode.r_outer_m <= geometry.max_radius_m):
                raise PIC2DValidationError("the cathode annulus must lie inside the plume box")
        if self.see is not None and self.see.enabled:
            raise PIC2DValidationError(
                "SEE emission is a v1.4 scaffold (yield model + wall diagnostics only); enabled=True is not implemented - fail closed"
            )
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
        } | ({} if self.neutral_inventory is None else {"neutral_inventory": self.neutral_inventory.to_dict()}) \
          | ({} if self.peak_debye_gate is None else {"peak_debye_gate": self.peak_debye_gate.to_dict()}) \
          | ({} if self.anomalous is None else {"anomalous": self.anomalous.to_dict()}) \
          | ({} if self.see is None else {"see": self.see.to_dict()}) \
          | ({} if self.cathode is None else {"cathode": self.cathode.to_dict()}) \
          | ({} if self.plume_boundary_gate is None else {"plume_boundary_gate": self.plume_boundary_gate.to_dict()})
        # (each key is present only when its option is on, so v1.0-v1.3 config identities are unchanged)

    @property
    def emission_peak_current_a(self) -> float:
        """Largest electron emission current any step can draw (sizes device buffers)."""

        if self.cathode is not None:
            return self.cathode.peak_current_a
        return self.injection.electron_current_a if self.injection is not None else 0.0

    @property
    def emission_temperature_ev(self) -> float:
        if self.cathode is not None:
            return self.cathode.electron_temperature_ev
        return self.injection.electron_temperature_ev if self.injection is not None else 0.0

    @property
    def initial_emission_rate_per_step(self) -> float:
        current = self.cathode.current_a if self.cathode is not None else (self.injection.electron_current_a if self.injection is not None else 0.0)
        return current * self.dt_s / (ELEMENTARY_CHARGE_C * self.macro_weight)

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
    """Fixed ledger (v1.x keys) plus the v2.0 momentum/plume tallies (stored as *extra* keys in checkpoints)."""

    return {key: 0.0 for key in CUMULATIVE_KEYS} | {key: 0.0 for key in MOMENTUM_KEYS}


# v2.0 momentum ledger and plume tallies.  They live in the *extra* part of the cumulative
# ledger (absent from v1.x checkpoints, read back with ``.get(key, 0.0)``), all in SI with the
# macro weight applied: ``pz_*`` are axial momenta in kg m/s, counts are macro-particles.
#   pz_impulse           sum over pushes of m W (v_z^+ - v_z^-)           (total force by E and B on the plasma)
#   pz_impulse_electric  sum over pushes of q E_z W dt                     (electric part; the rest is q v x B)
#   pz_collisions        m_e W dv_z in MCC and Bohm scattering            (momentum handed to the neutral gas)
#   pz_born              momentum of ionisation products                  (secondary electrons + ions)
#   pz_injected          momentum of emitted electrons
#   pz_exit_*            momentum carried through the far-field boundary  (the beam)
#   pz_wall_* / pz_anode_*  momentum deposited on the dielectric (incl. front face) / anode
#   body_face_*          wall hits on the thruster front face (r >= r_exit at the exit plane; not recycled)
#   ionizations_plume    ionisation events downstream of the exit plane (consume effused atoms, not inventory)
MOMENTUM_KEYS = (
    "pz_impulse", "pz_impulse_electric", "pz_collisions", "pz_born", "pz_injected",
    "pz_exit_electrons", "pz_exit_ions", "pz_wall_electrons", "pz_wall_ions", "pz_anode_electrons", "pz_anode_ions",
    "body_face_electrons", "body_face_ions", "ionizations_plume",
)
# v2.0: the continuity-rule emission rate is dynamical state (checkpointed as an extra ledger scalar)
CATHODE_RATE_KEY = "cathode_rate_per_step"
# v2.0 plume histograms: ion current per solid angle in 1 degree bins from the exit-aperture centre and
# the ion energy distribution at the far-field boundary (256 bins over [0, iedf_max_ev])
THETA_BINS = 90
IEDF_BINS = 256


def momentum_z_kg_m_s(species: Species2D, particles: ParticleArrays) -> float:
    """Represented axial momentum ``sum W m v_z`` (classical; the ledger tallies the same quantity)."""

    return float(np.sum(particles.vz_m_per_s)) * species.mass_kg * species.macro_weight


@dataclass(slots=True)
class StepTally:
    poisson_iterations: int
    max_omega_pe_dt: float
    max_electron_speed_m_per_s: float
    electron_count: int
    ion_count: int


@dataclass(frozen=True, slots=True)
class PeakDebyeGateConfig:
    """v1.4 runtime gate on the *peak-density node* (literature review, blocker 1).

    At every series record the electron density and temperature are evaluated on
    the node with the highest electron density; ``max(dr, dz) / lambda_D`` there must
    not exceed ``max_cells_per_debye`` (fail-closed ``PIC2DStabilityError``).  The
    a-priori gate ``StabilityLimits.max_cell_debye_ratio`` is evaluated once at a
    reference density; this one follows the plasma (Brandt et al. 2016 ran 2
    lambda_D per cell at the peak; the v1.3 development plateau reached 3).  Nodes
    holding fewer than ``min_macro_particles_at_peak`` electrons give an unreliable
    temperature; the gate is then recorded but not enforced.  ``dense_fraction``
    defines the "densest cells" (n >= dense_fraction * n_peak) whose density-weighted
    T_e is recorded for the grid-heating triad.
    """

    max_cells_per_debye: float
    min_macro_particles_at_peak: int = 16
    dense_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not isfinite(self.max_cells_per_debye) or self.max_cells_per_debye <= 0.0:
            raise PIC2DValidationError("max_cells_per_debye must be positive")
        if isinstance(self.min_macro_particles_at_peak, bool) or not isinstance(self.min_macro_particles_at_peak, int) or self.min_macro_particles_at_peak < 1:
            raise PIC2DValidationError("min_macro_particles_at_peak must be a positive integer")
        if not 0.0 < self.dense_fraction <= 1.0:
            raise PIC2DValidationError("dense_fraction must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {"max_cells_per_debye": self.max_cells_per_debye, "min_macro_particles_at_peak": self.min_macro_particles_at_peak,
                "dense_fraction": self.dense_fraction}


def peak_node_debye(
    masks: MeshMasks, config: "PIC2DConfig", weight: np.ndarray, vr: np.ndarray, vt: np.ndarray, vz: np.ndarray,
    v2: np.ndarray, *, dense_fraction: float = 0.5, min_particles: int = 16,
) -> dict[str, Any]:
    """Peak-node density, temperature, Debye length and cells per Debye length from node moments.

    The moments are bilinear node sums over the electrons at their current positions of
    macro-particle weight (1 each), velocity components and speed squared; the density is
    ``weight * W / shape_volume``.  T_e is the thermal temperature
    ``m_e (<v^2> - |<v>|^2) / (3 e)`` at the node.  The peak node is the densest node
    holding at least ``min_particles`` macro-electrons (single-particle noise on the small
    axis nodes would otherwise dominate the argmax; the unrestricted maximum is reported
    as ``raw_peak``).  Also returns the density-weighted T_e over the densest nodes
    (n >= dense_fraction * n_peak).
    """

    plasma = masks.plasma_node
    with np.errstate(invalid="ignore", divide="ignore"):
        n_e = np.where(plasma, weight * config.macro_weight / masks.shape_volume_m3, 0.0)
        mean_v2 = np.where(weight > 0.0, v2 / np.maximum(weight, 1e-300), 0.0)
        drift2 = np.where(weight > 0.0, (vr**2 + vt**2 + vz**2) / np.maximum(weight, 1e-300) ** 2, 0.0)
        t_e = np.maximum(mean_v2 - drift2, 0.0) * ELECTRON_MASS_KG / (3.0 * EV_J)
    raw_flat = int(np.argmax(n_e))
    qualified = np.where(weight >= float(min_particles), n_e, -1.0)
    flat = int(np.argmax(qualified)) if np.any(qualified >= 0.0) else raw_flat
    i, j = np.unravel_index(flat, n_e.shape)
    n_peak = float(n_e[i, j])
    t_peak = float(t_e[i, j])
    particles = float(weight[i, j])
    grid = masks.grid
    cell = max(grid.dr_m, grid.dz_m)
    debye: float | None
    if n_peak > 0.0 and t_peak > 0.0:
        debye = sqrt(EPSILON_0_F_PER_M * t_peak * EV_J / (n_peak * ELEMENTARY_CHARGE_C**2))
        cells_per_debye = cell / debye
    else:
        debye, cells_per_debye = None, 0.0      # no plasma at the peak: lambda_D undefined (None keeps the JSON finite)
    dense = n_e >= dense_fraction * n_peak if n_peak > 0.0 else np.zeros_like(plasma)
    dense_weight = n_e[dense]
    t_dense = float(np.sum(t_e[dense] * dense_weight) / dense_weight.sum()) if dense_weight.sum() > 0.0 else 0.0
    return {
        "node": [int(i), int(j)],
        "r_m": float(i * grid.dr_m),
        "z_m": float(grid.geometry.z_min_m + j * grid.dz_m),
        "n_e_peak_per_m3": n_peak,
        "t_e_peak_ev": t_peak,
        "macro_particles_at_peak": particles,
        "debye_length_m": debye,
        "cells_per_debye": cells_per_debye,
        "dz_per_debye": grid.dz_m / debye if debye is not None else 0.0,
        "dr_per_debye": grid.dr_m / debye if debye is not None else 0.0,
        "t_e_dense_ev": t_dense,
        "dense_node_count": int(dense.sum()),
        "min_particles_for_peak": int(min_particles),
        "raw_peak": {"node": [int(k) for k in np.unravel_index(raw_flat, n_e.shape)], "n_e_per_m3": float(n_e.flat[raw_flat]),
                     "macro_particles": float(weight.flat[raw_flat])},
    }


class DiagnosticAccumulator:
    """Time-window sums of node maps and boundary fluxes (CPU numpy)."""

    def __init__(self, masks: MeshMasks, iedf_max_ev: float = 450.0) -> None:
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
        # v2.0 plume block: far-field side boundary (r = R_plume) flux per axial cell, ion current per
        # solid angle from the aperture centre, ion energy distribution at the far-field boundary
        self.side_ions = np.zeros(nz)
        self.side_electrons = np.zeros(nz)
        self.theta_ions = np.zeros(THETA_BINS)
        self.iedf_ions = np.zeros(IEDF_BINS)
        self.iedf_max_ev = float(iedf_max_ev)

    def reset(self) -> None:
        self.__init__(self.masks, self.iedf_max_ev)

    # v2.0 frame recorder: the raw window sums, so that an interval [a, b] inside the window is
    # recovered exactly as the difference of two cumulative snapshots (sums are additive)
    SUM_KEYS = (
        "n_e", "n_i", "phi", "e_weight", "e_vr", "e_vt", "e_vz", "e_v2", "ionization", "wall_electrons", "wall_ions",
        "wall_electron_energy_j", "wall_ion_energy_j", "exit_ions", "exit_electrons", "side_ions", "side_electrons",
        "theta_ions", "iedf_ions",
    )

    def raw_sums(self) -> dict[str, np.ndarray]:
        out = {key: np.asarray(getattr(self, key)).copy() for key in self.SUM_KEYS}
        out["steps"] = np.array([self.steps], dtype=np.int64)
        return out

    @classmethod
    def from_sums(cls, masks: MeshMasks, sums: Mapping[str, np.ndarray], iedf_max_ev: float = 450.0) -> "DiagnosticAccumulator":
        acc = cls(masks, iedf_max_ev)
        for key in cls.SUM_KEYS:
            setattr(acc, key, np.asarray(sums[key], dtype=np.float64).copy())
        acc.steps = int(np.asarray(sums["steps"]).reshape(-1)[0])
        return acc

    def record_exit(self, is_electron: bool, r_m: np.ndarray, z_m: np.ndarray, energy_ev: np.ndarray) -> None:
        """Bin far-field crossings (positions after the push) into the exit histograms."""

        grid = self.masks.grid
        geometry = grid.geometry
        if r_m.size == 0:
            return
        through_plane = z_m >= geometry.domain_z_max_m
        i = np.clip((r_m[through_plane] / grid.dr_m).astype(np.int64), 0, grid.radial_cells - 1)
        np.add.at(self.exit_electrons if is_electron else self.exit_ions, i, 1.0)
        side = ~through_plane
        if np.any(side):
            j = np.clip(((z_m[side] - geometry.z_min_m) / grid.dz_m).astype(np.int64), 0, grid.axial_cells - 1)
            np.add.at(self.side_electrons if is_electron else self.side_ions, j, 1.0)
        if not is_electron:
            theta = np.degrees(np.arctan2(r_m, np.maximum(z_m - geometry.z_max_m, 0.0)))
            np.add.at(self.theta_ions, np.clip((theta * THETA_BINS / 90.0).astype(np.int64), 0, THETA_BINS - 1), 1.0)
            e_bin = np.clip((energy_ev * IEDF_BINS / self.iedf_max_ev).astype(np.int64), 0, IEDF_BINS - 1)
            np.add.at(self.iedf_ions, e_bin, 1.0)

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
        # v2.0: the side boundary r = R_plume (area 2 pi R dz per axial cell), the ion current per solid
        # angle about the aperture centre (bin solid angle 2 pi (cos a - cos b)) and the IEDF
        r_outer = grid.geometry.max_radius_m
        side_area = 2.0 * pi * r_outer * dz
        theta_edges = np.radians(np.linspace(0.0, 90.0, THETA_BINS + 1))
        solid_angle = 2.0 * pi * (np.cos(theta_edges[:-1]) - np.cos(theta_edges[1:]))
        iedf_edges = np.linspace(0.0, self.iedf_max_ev, IEDF_BINS + 1)
        return {
            "n_e_per_m3": self.n_e / steps,
            "side_ion_current_density_a_per_m2": self.side_ions * electron_weight * ELEMENTARY_CHARGE_C / (side_area * window_s),
            "side_electron_current_density_a_per_m2": self.side_electrons * electron_weight * ELEMENTARY_CHARGE_C / (side_area * window_s),
            "plume_ion_current_per_sr_a": self.theta_ions * electron_weight * ELEMENTARY_CHARGE_C / (solid_angle * window_s),
            "plume_theta_edges_deg": np.degrees(theta_edges),
            "plume_ion_counts_per_theta": self.theta_ions.copy(),
            "iedf_ion_counts": self.iedf_ions.copy(),
            "iedf_edges_ev": iedf_edges,
            "sample_count_e": self.e_weight.copy(),
            "window_s": np.array([window_s]),
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
        self.diagnostics = DiagnosticAccumulator(masks, iedf_max_ev=iedf_max_ev(config))
        self.diagnostic_generation = 0     # v2.0.2: incremented by every reset_diagnostics (window bridging)
        self.quantum_c = ELEMENTARY_CHARGE_C * config.macro_weight
        self.last_tally: StepTally | None = None
        self._last_charge_maps: tuple[np.ndarray, np.ndarray] | None = None
        # v2.0: two-zone neutral density shape (1 in the channel, free-molecular cone in the plume)
        self.neutral_shape_cell = neutral_shape_cells(masks)
        self.emission_rate_per_step = config.initial_emission_rate_per_step

    # -- state exchange -------------------------------------------------
    def load_state(self, state: SimulationState) -> None:
        self.state = state.copy()
        self._last_charge_maps = None
        self.emission_rate_per_step = float(state.cumulative.get(CATHODE_RATE_KEY, self.config.initial_emission_rate_per_step))

    def export_state(self) -> SimulationState:
        assert self.state is not None
        return self.state.copy()

    def set_neutral_scale(self, scale: float) -> None:
        """v1.3: real-collision frequency factor ``n_g / n_g0`` (null ceiling fixed at ``n_g0``)."""

        if self.mcc is None:
            raise PIC2DValidationError("neutral scale requires MCC")
        self.mcc.set_neutral_scale(scale)

    def set_emission_rate(self, rate_per_step: float) -> None:
        """v2.0: cathode emission rate (macro-electrons per step) for the coming steps (continuity rule)."""

        assert self.state is not None
        if not isfinite(rate_per_step) or rate_per_step < 0.0:
            raise PIC2DValidationError("emission rate must be finite and non-negative")
        self.emission_rate_per_step = float(rate_per_step)
        self.state.cumulative[CATHODE_RATE_KEY] = self.emission_rate_per_step

    @property
    def step_index(self) -> int:
        assert self.state is not None
        return self.state.step

    @property
    def time_s(self) -> float:
        assert self.state is not None
        return self.state.time_s

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
            "momentum_z_electrons": momentum_z_kg_m_s(self.electron, state.electrons),
            "momentum_z_ions": momentum_z_kg_m_s(self.ion, state.ions),
            "surface_charge_c": float(state.surface_charge_c.sum()), "phi_v": state.phi_v.copy(),
            "surface_charge_map_c": state.surface_charge_c.copy(),
            "cumulative": dict(state.cumulative),
        }

    def charge_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """v2.0: node charge maps (electrons, ions) of the last field solve (plume-boundary sample).

        The same sampling moment as the Warp backend (the deposit that produced ``state.phi_v``);
        before the first step (or after ``load_state``) the maps are deposited at the current positions.
        """

        assert self.state is not None
        if self._last_charge_maps is None:
            fixed = self.config.fixed_point_deposition
            return (kernels.deposit_node_charge(self.masks, self.electron, self.state.electrons, fixed_point=fixed),
                    kernels.deposit_node_charge(self.masks, self.ion, self.state.ions, fixed_point=fixed))
        return self._last_charge_maps

    def peak_node_sample(self) -> dict[str, Any]:
        """v1.4: instantaneous electron density and temperature moments on the nodes (for the peak-node Debye gate)."""

        assert self.state is not None
        electrons = self.state.electrons
        masks = self.masks
        shape = masks.grid.node_shape
        if electrons.count:
            moments = [kernels.deposit_node_moment(masks, electrons, values) for values in (
                np.ones(electrons.count), electrons.vr_m_per_s, electrons.vt_m_per_s, electrons.vz_m_per_s, electrons.speed_squared())]
        else:
            moments = [np.zeros(shape) for _ in range(5)]
        gate = self.config.peak_debye_gate
        return peak_node_debye(masks, self.config, *moments, dense_fraction=gate.dense_fraction if gate is not None else 0.5,
                               min_particles=gate.min_macro_particles_at_peak if gate is not None else 16)

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
        self._last_charge_maps = (q_e, q_i)
        volume_charge = q_e + q_i
        source = volume_charge * masks.charge_to_source + state.surface_charge_c
        result = self.poisson.solve(source, config.potentials, initial_phi_v=state.phi_v)
        phi = result.phi_v
        e_r, e_z = electric_field_nodes(masks, phi)

        if accumulate:
            self._accumulate_maps(q_e, q_i, phi, electrons)

        max_speed = 0.0
        field_work = 0.0
        cumulative = state.cumulative
        add = lambda key, value: cumulative.__setitem__(key, cumulative.get(key, 0.0) + float(value))  # noqa: E731
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
            # v2.0 momentum ledger: the total impulse (E and B) and its electric part q E_z dt
            mass_weight = species.mass_kg * species.macro_weight
            add("pz_impulse", mass_weight * float(np.sum(vz - particles.vz_m_per_s)))
            add("pz_impulse_electric", species.charge_c * species.macro_weight * species_dt * float(np.sum(ez)))
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

        if config.anomalous is not None and electrons.count:
            # v1.4 hook: Bohm-type isotropic scattering at nu_an = alpha omega_ce(x) (speed preserved)
            br = kernels.gather_nodes(grid, self.field.b_r_t, electrons.r_m, electrons.z_m)
            bz = kernels.gather_nodes(grid, self.field.b_z_t, electrons.r_m, electrons.z_m)
            rng_an = np.random.default_rng([config.seed, state.step, 3])
            vr_s, vt_s, vz_s, hits = apply_bohm_scattering(
                config.anomalous.alpha, electrons.vr_m_per_s, electrons.vt_m_per_s, electrons.vz_m_per_s, np.hypot(br, bz), dt, rng_an,
            )
            add("pz_collisions", self.electron.mass_kg * config.macro_weight * float(np.sum(vz_s - electrons.vz_m_per_s)))
            electrons = ParticleArrays(electrons.r_m, electrons.z_m, vr_s, vt_s, vz_s)
            state.cumulative["anomalous"] = state.cumulative.get("anomalous", 0.0) + hits

        if self.mcc is not None and electrons.count:
            rng = np.random.default_rng([config.seed, state.step, 1])
            shape = None
            if masks.has_plume:
                ci, cj, _, _ = cell_index(grid, electrons.r_m, electrons.z_m)
                shape = self.neutral_shape_cell[ci, cj]
            mcc_result = self.mcc.apply(electrons, dt, rng, density_shape=shape)
            add("pz_collisions", self.electron.mass_kg * config.macro_weight
                * float(np.sum(mcc_result.electrons.vz_m_per_s - electrons.vz_m_per_s)))
            electrons = mcc_result.electrons
            tally = mcc_result.tally
            if mcc_result.new_electrons.count:
                electrons = electrons.append(mcc_result.new_electrons)
                ions = ions.append(mcc_result.new_ions)
                state.cumulative["ke_born_ions_j"] += kernels.kinetic_energy_j(self.ion, mcc_result.new_ions)
                add("pz_born", momentum_z_kg_m_s(self.ion, mcc_result.new_ions) + momentum_z_kg_m_s(self.electron, mcc_result.new_electrons))
                if masks.has_plume:
                    add("ionizations_plume", int(np.count_nonzero(mcc_result.new_ions.z_m >= grid.geometry.z_max_m)))
                if accumulate:
                    self.diagnostics.ionization += kernels.deposit_node_moment(
                        masks, mcc_result.new_ions, np.ones(mcc_result.new_ions.count)
                    )
            state.cumulative["ionizations"] += tally.ionization
            state.cumulative["excitations"] += tally.excitation
            state.cumulative["elastic"] += tally.elastic
            state.cumulative["inelastic_loss_j"] += tally.inelastic_energy_loss_j

        if self.emission_rate_per_step > 0.0:
            injected, state.injection_carry = self._inject(state.step, state.injection_carry)
            if injected.count:
                electrons = electrons.append(injected)
                state.cumulative["injected_electrons"] += injected.count
                state.cumulative["ke_injected_j"] += kernels.kinetic_energy_j(self.electron, injected)
                add("pz_injected", momentum_z_kg_m_s(self.electron, injected))

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
        mass_weight = species.mass_kg * species.macro_weight
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
            momentum_key = f"pz_{name}_{label}"
            state.cumulative[momentum_key] = state.cumulative.get(momentum_key, 0.0) + mass_weight * float(moved.vz_m_per_s[mask].sum())
            if code == kernels.BOUNDARY_WALL:
                charge = np.full(count, species.charge_c * species.macro_weight)
                state.surface_charge_c += kernels.wall_surface_deposit(
                    self.masks, moved.r_m[mask], moved.z_m[mask], charge,
                    fixed_point=self.config.fixed_point_deposition, quantum_c=self.quantum_c,
                )
                if self.masks.has_plume:
                    # v2.0: hits on the thruster front face (outside the exit lip) are not channel-wall hits
                    face_key = f"body_face_{label}"
                    state.cumulative[face_key] = state.cumulative.get(face_key, 0.0) + int(
                        np.count_nonzero(moved.r_m[mask] >= grid.geometry.exit_radius_m))
                if accumulate:
                    j = np.clip(((moved.z_m[mask] - grid.geometry.z_min_m) / grid.dz_m).astype(np.int64), 0, grid.axial_cells - 1)
                    target = self.diagnostics.wall_electrons if is_electron else self.diagnostics.wall_ions
                    energy_target = self.diagnostics.wall_electron_energy_j if is_electron else self.diagnostics.wall_ion_energy_j
                    np.add.at(target, j, 1.0)
                    np.add.at(energy_target, j, ke[mask])
            elif code == kernels.BOUNDARY_EXIT and accumulate:
                self.diagnostics.record_exit(is_electron, moved.r_m[mask], moved.z_m[mask], ke[mask] / (species.macro_weight * EV_J))

    def _inject(self, step: int, carry: float) -> tuple[ParticleArrays, float]:
        config = self.config
        expected = self.emission_rate_per_step + carry
        count = int(np.floor(expected))
        carry = expected - count
        if count == 0:
            return ParticleArrays.empty(), carry
        rng = np.random.default_rng([config.seed, step, 2])
        u = rng.random((7, count))
        if config.cathode is not None:
            return cathode_sample(config, u), carry
        return injection_sample(config, self.masks, u), carry

    def diagnostic_arrays(self) -> dict[str, np.ndarray]:
        return self.diagnostics.to_arrays(self.config.macro_weight, self.config.dt_s)

    def diagnostic_sums(self) -> dict[str, np.ndarray]:
        """v2.0 frame recorder: the cumulative window sums (additive; see DiagnosticAccumulator.raw_sums)."""

        return self.diagnostics.raw_sums()

    def far_field_window_sums(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """v2.0.2 plume-boundary gate: far-field rows of the window sums ``sum_t n_e``, ``sum_t n_i`` (m^-3 x steps),
        the accumulated step count and the reset generation (same accumulation as ``diagnostic_sums``)."""

        far = self.masks.far_field_node
        return self.diagnostics.n_e[far].copy(), self.diagnostics.n_i[far].copy(), int(self.diagnostics.steps), self.diagnostic_generation

    def surface_charge_map(self) -> np.ndarray:
        assert self.state is not None
        return self.state.surface_charge_c.copy()

    def reset_diagnostics(self) -> None:
        self.diagnostics.reset()
        self.diagnostic_generation += 1


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


def cathode_sample(config: PIC2DConfig, u: np.ndarray) -> ParticleArrays:
    """Map uniforms ``u`` (shape (7, N)) to cathode-annulus electrons (v2.0).

    Position uniform in volume over the annulus (``r = sqrt(r_in^2 + u (r_out^2 - r_in^2))``,
    ``z`` uniform), velocity an isotropic Maxwellian at the cathode temperature (four
    uniforms, Box-Muller as ``maxwellian_velocity``).  Shared by both backends.
    """

    cathode = config.cathode
    assert cathode is not None
    r = np.sqrt(cathode.r_inner_m**2 + u[0] * (cathode.r_outer_m**2 - cathode.r_inner_m**2))
    z = cathode.z_start_m + u[1] * (cathode.z_end_m - cathode.z_start_m)
    temperature_k = cathode.electron_temperature_ev * EV_J / 1.380649e-23
    vr, vt, vz = maxwellian_velocity(ELECTRON_MASS_KG, temperature_k, u[2:6])
    return ParticleArrays(r, z, vr, vt, vz)


def iedf_max_ev(config: PIC2DConfig) -> float:
    """Upper edge of the far-field ion energy histogram: 1.5 x the anode potential (at least 10 eV)."""

    return max(1.5 * abs(config.potentials.anode_v - config.potentials.exit_v), 10.0)


def plume_neutral_shape(geometry: Grid2D | Any, r_m: np.ndarray, z_m: np.ndarray) -> np.ndarray:
    """v2.0 two-zone neutral density shape ``n_g(r, z) / n_g,channel``.

    Channel (``z < z_exit``): 1 (the v1.3/v1.4 uniform inventory).  Plume: free-molecular
    (Knudsen) effusion from the exit aperture treated as a cosine-law point source of the
    channel's effusion flux ``Phi = n_g v_bar A / 4``: the flux density at distance ``rho`` and
    angle ``theta`` from the aperture axis is ``Phi cos(theta) / (pi rho^2)``, the mean speed
    of the effusing atoms is ``3 pi v_bar / 8``, so ``n / n_g = 2 A cos(theta) / (3 pi^2 rho^2)``,
    capped at 1/2 (the outgoing half-Maxwellian at the aperture).  Ion-neutral collisions
    stay off (no hash-bound Xe+-Xe table in the repository); this field only sets the
    electron-neutral MCC rate in the plume.
    """

    geom = geometry.geometry if isinstance(geometry, Grid2D) else geometry
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    shape = np.ones(np.broadcast(r, z).shape, dtype=np.float64)
    if not geom.has_plume:
        return shape
    dz_exit = z - geom.z_max_m
    plume = dz_exit >= 0.0
    rho2 = r**2 + dz_exit**2
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = np.where(rho2 > 0.0, dz_exit / np.sqrt(rho2), 1.0)
        cone = 2.0 * pi * geom.exit_radius_m**2 * cos_theta / (3.0 * pi**2 * np.where(rho2 > 0.0, rho2, np.inf))
    return np.where(plume, np.minimum(0.5, np.maximum(cone, 0.0)), shape)


def neutral_shape_cells(masks: MeshMasks) -> np.ndarray:
    """Cell-centred neutral density shape (``(nr, nz)``; zero on non-plasma cells)."""

    grid = masks.grid
    r_mid = 0.5 * (grid.r_m[:-1] + grid.r_m[1:])
    z_mid = 0.5 * (grid.z_m[:-1] + grid.z_m[1:])
    shape = plume_neutral_shape(grid, r_mid[:, None], z_mid[None, :])
    return np.where(masks.plasma_cell, shape, 0.0)


def seed_plasma_state(config: PIC2DConfig, masks: MeshMasks) -> SimulationState:
    """Uniform quasi-neutral seed plasma (or empty) as the initial state."""

    grid = masks.grid
    electrons = ParticleArrays.empty()
    ions = ParticleArrays.empty()
    if config.seed_plasma is not None and config.seed_plasma.density_per_m3 > 0.0:
        geometry = grid.geometry
        channel_only = config.seed_plasma.region == "channel" and geometry.has_plume
        volume = float(masks.channel_volume_m3) if channel_only else masks.plasma_volume_m3
        count = int(round(config.seed_plasma.density_per_m3 * volume / config.macro_weight))
        rng = np.random.default_rng([config.seed, 0, 3])
        r_list: list[np.ndarray] = []
        z_list: list[np.ndarray] = []
        accepted = 0
        r_box = geometry.exit_radius_m if channel_only else geometry.max_radius_m
        z_span = geometry.channel_length_m if channel_only else geometry.length_m
        while accepted < count:
            batch = max(1024, 2 * (count - accepted))
            r = r_box * np.sqrt(rng.random(batch))
            z = geometry.z_min_m + z_span * rng.random(batch)
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
    # v1.4: peak-node Debye sample (always recorded; gated when the config has a peak_debye_gate)
    peak_node: dict[str, Any] | None = None
    # v2.0: momentum ledger, thrust estimates and plume-boundary sample (plume geometries only)
    momentum: dict[str, Any] | None = None
    plume: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "time_s": self.time_s, "electrons": self.electrons, "ions": self.ions,
            "phi_mean_v": self.phi_mean_v, "phi_min_v": self.phi_min_v, "phi_max_v": self.phi_max_v,
            "kinetic_electron_j": self.kinetic_electron_j, "kinetic_ion_j": self.kinetic_ion_j,
            "field_energy_j": self.field_energy_j, "surface_charge_c": self.surface_charge_c,
            "peak_omega_pe_dt": self.peak_omega_pe_dt, "poisson_iterations": self.poisson_iterations,
            "currents_a": dict(self.currents_a), "ledger": dict(self.ledger),
            "neutral": None if self.neutral is None else dict(self.neutral),
            "peak_node": None if self.peak_node is None else dict(self.peak_node),
        } | ({} if self.momentum is None else {"momentum": dict(self.momentum)}) \
          | ({} if self.plume is None else {"plume": dict(self.plume)})


# v2.0.2 plume-boundary gate: trailing window of ACCUMULATED steps over which the far-field charge statistic is
# averaged (the plume protocol's 400 000-step / 0.6 us averaging window = 20 frames) and the sample-size floor in
# accumulated macro-particle weight per node over that window (see PlumeBoundaryGateConfig for the derivation:
# 32 independent beam-ion crossings of a node x 2000 steps per crossing).  Also the recording values when the gate is off.
PLUME_GATE_WINDOW_STEPS = 400_000
PLUME_GATE_MIN_ACCUMULATED_MACRO_PARTICLES_PER_NODE = 64_000.0


@dataclass(frozen=True, slots=True)
class PlumeBoundaryGateConfig:
    """v2.0 plume-boundary sanity gate (fail-closed), v2.0.2 window-averaged form.

    The far-field nodes are Dirichlet at the reference potential, so the check is on
    *charge pile-up*: a net charge density at the far-field nodes larger than
    ``max_charge_fraction`` of the peak electron density in the domain means a sheath is
    forming on the box boundary, i.e. the plume box is too small for the beam to leave
    quasi-neutrally (Brandt et al. 2016 found a 20 x 5 mm box "still too small"; the
    outer-boundary condition changed their plume current ratios).  Enforced after
    ``enforce_after_s`` so the seed plasma's first transit does not trip it.

    Gate quantity (v2.0.2): ``max over resolved far-field nodes of |<n_i> - <n_e>| / <n_e,peak>``
    where ``<.>`` is the time average over the trailing window of at least ``window_steps``
    accumulated steps, read from the SAME window accumulators that produce ``maps.npz`` and the
    frames (``sum_t n_e``, ``sum_t n_i`` on the far-field nodes; the backend reads them at the
    series-record host sync, never per step) and bridged across the runner's window resets by a
    host-side carry, so the window is continuous.  The denominator is the mean over the window's
    series records of the instantaneous peak electron density.  A far-field node is *resolved*
    when its accumulated macro-particle weight over the window (electrons + ions, bilinear
    weights, summed over the accumulated steps: ``(sum_t n_e + sum_t n_i) V_node / W``) is at
    least ``min_accumulated_macro_particles_per_node``.

    Why a window and this floor.  v2.0 read a SINGLE-deposit statistic: the axis corner node of
    the far plane has a bilinear shape volume ``pi dr^2 dz / 6`` (6.5e-14 m^3 on the 50 um plume
    grid), so ONE macro-ion (W = 6e4) there reads 9.2e17 m^-3 = 0.4 of a 2.3e18 peak; plume
    attempt 6 (2026-09-04) was stopped by 0.66 macro-ions and no electrons on that node (0.259 of
    the peak) while the interval-averaged far-field charge fraction was 0.03.  v2.0.1 kept the
    single-deposit statistic and added a floor of 32 macro-particles PER DEPOSIT; attempt 7
    showed that floor is unreachable at far-field densities (0.01-20 macro-particles per node,
    0 resolved nodes in all 4601 armed records): the gate no longer false-fired but could not
    fire at all.  The window average is the statistic the attempt-6/7 diagnoses actually used
    (0.030-0.035 max over the far-field nodes).  Floor derivation: an accumulated weight ``A``
    is particle-steps, and a beam ion (~15 km/s at the far plane, attempt 7: flux-weighted
    <v_z> 14.8 km/s, IEDF 10 % quantile 13 km/s) stays on a 50 um node for ``tau`` = 50 um /
    (15 km/s x 1.5 ps) = 2000 steps (the ~10 series intervals per corner-node crossing seen in
    the attempt-6 log), so the number of independent samples is ``N_eff = A / (w tau) >= A / tau``
    (bilinear weight w <= 1).  ``A >= 32 tau = 64 000`` therefore guarantees >= 32 independent
    beam-ion crossings (the peak-node Debye gate's 32-particle convention, relative shot noise
    <= 18 %) for every particle at least as fast as the beam; electrons (60x faster) only add
    samples.  Over the 400 000-step window this is a mean occupancy of 0.16 macro-particles per
    node: on the attempt-7 window maps it resolves 121 of the 481 far-field nodes (the far plane out
    to r = 6.7 mm except the axis corner node at occupancy 0.10, which its resolved neighbour (1, nz)
    covers at 0.0339 of the peak; attempt 6: 77 nodes, 0.0249); a genuine sheath at the 0.25 threshold puts >= 0.73 macro-ions
    on the corner node and >= 4.4 on its neighbour at all times, i.e. is resolved everywhere but
    the corner.  The unrestricted window statistic (all far-field nodes) and the v2.0.1
    single-deposit statistic are recorded alongside; the threshold (0.25, 7x the attempt-7
    window value) and the arming time are unchanged.
    """

    max_charge_fraction: float = 0.25
    enforce_after_s: float = 0.0
    window_steps: int = PLUME_GATE_WINDOW_STEPS
    min_accumulated_macro_particles_per_node: float = PLUME_GATE_MIN_ACCUMULATED_MACRO_PARTICLES_PER_NODE

    def __post_init__(self) -> None:
        if not isfinite(self.max_charge_fraction) or self.max_charge_fraction <= 0.0:
            raise PIC2DValidationError("max_charge_fraction must be positive")
        if not isfinite(self.enforce_after_s) or self.enforce_after_s < 0.0:
            raise PIC2DValidationError("enforce_after_s must be non-negative")
        if isinstance(self.window_steps, bool) or not isinstance(self.window_steps, int) or self.window_steps < 1:
            raise PIC2DValidationError("window_steps must be a positive integer")
        floor = self.min_accumulated_macro_particles_per_node
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not isfinite(floor) or floor <= 0.0:
            raise PIC2DValidationError("min_accumulated_macro_particles_per_node must be a positive finite number")
        object.__setattr__(self, "min_accumulated_macro_particles_per_node", float(floor))

    def to_dict(self) -> dict[str, float | int]:
        return {"max_charge_fraction": self.max_charge_fraction, "enforce_after_s": self.enforce_after_s,
                "window_steps": self.window_steps,
                "min_accumulated_macro_particles_per_node": self.min_accumulated_macro_particles_per_node}


class FarFieldChargeWindow:
    """v2.0.2: trailing-window far-field charge statistic from the diagnostic accumulators (host side).

    At every series record (an existing host sync) the backend hands over the far-field rows of
    its window sums ``sum_t n_e``, ``sum_t n_i``, the accumulated step count and a reset
    generation counter.  Cumulative totals bridge the runner's accumulator resets (a generation
    change carries the last reading before the reset, which in the runner IS the reset boundary
    because every window reset follows a series record), and a ring of totals per record turns
    the trailing window of at least ``window_steps`` accumulated steps into an exact difference
    of two totals - the frame recorder's construction (``frames.interval_maps``).  The window is
    ``min(window_steps, all accumulated history)`` until enough history exists; ``complete`` says
    whether it reached ``window_steps``.  Memory: far-field nodes only (481 on the plume grid),
    ``2 ceil(window_steps / series_interval) + 2`` entries.
    """

    def __init__(self, masks: MeshMasks, macro_weight: float, window_steps: int, series_interval_steps: int, floor: float) -> None:
        self.far = masks.far_field_node
        self.volume = np.asarray(masks.shape_volume_m3[self.far], dtype=np.float64)
        self.macro_weight = float(macro_weight)
        self.window_steps = int(window_steps)
        self.floor = float(floor)
        self.maxlen = 2 * (-(-self.window_steps // max(int(series_interval_steps), 1))) + 2
        # ring entries: (step, cumulative accumulated steps, cumulative peak sum, record index, cumulative sum_e, cumulative sum_i)
        self._ring: deque[tuple[int, int, float, int, np.ndarray, np.ndarray]] = deque(maxlen=self.maxlen)
        self._carry_e = np.zeros(int(self.far.sum()))
        self._carry_i = np.zeros(int(self.far.sum()))
        self._carry_steps = 0
        self._last: tuple[np.ndarray, np.ndarray, int, int] | None = None
        self._peak_total = 0.0
        self._records = 0

    def reset(self, reading: tuple[np.ndarray, np.ndarray, int, int], step: int) -> None:
        """Forget the history (fresh start / loaded checkpoint) and seed the ring with the current totals."""

        self._ring.clear()
        self._carry_e[...] = 0.0
        self._carry_i[...] = 0.0
        self._carry_steps = 0
        self._peak_total = 0.0
        self._records = 0
        self._last = reading
        n_e, n_i, steps, _ = reading
        self._ring.append((int(step), int(steps), 0.0, 0, np.asarray(n_e, dtype=np.float64).copy(), np.asarray(n_i, dtype=np.float64).copy()))

    def update(self, reading: tuple[np.ndarray, np.ndarray, int, int], step: int, peak_now: float) -> dict[str, Any]:
        n_e = np.asarray(reading[0], dtype=np.float64)
        n_i = np.asarray(reading[1], dtype=np.float64)
        steps, generation = int(reading[2]), int(reading[3])
        if self._last is not None and generation != self._last[3]:
            # the accumulators were reset since the previous record: carry the last reading (the completed window)
            self._carry_e += np.asarray(self._last[0], dtype=np.float64)
            self._carry_i += np.asarray(self._last[1], dtype=np.float64)
            self._carry_steps += int(self._last[2])
        self._last = reading
        self._records += 1
        self._peak_total += float(peak_now)
        total_e = self._carry_e + n_e
        total_i = self._carry_i + n_i
        total_steps = self._carry_steps + steps
        # the newest ring entry at least window_steps of accumulation back (else the oldest available)
        base = self._ring[0] if self._ring else (int(step), total_steps, self._peak_total, self._records, total_e, total_i)
        for entry in reversed(self._ring):
            if total_steps - entry[1] >= self.window_steps:
                base = entry
                break
        window_steps = total_steps - base[1]
        records = self._records - base[3]
        out: dict[str, Any] = {
            "window_steps": int(window_steps), "window_records": int(max(records, 0)), "window_start_step": int(base[0]),
            "window_complete": bool(window_steps >= self.window_steps), "window_steps_required": self.window_steps,
            "min_accumulated_macro_particles_per_node": self.floor,
        }
        peak_mean = (self._peak_total - base[2]) / records if records > 0 else float(peak_now)
        out["peak_electron_density_window_per_m3"] = float(peak_mean)
        if window_steps > 0 and self.far.any():
            mean_e = (total_e - base[4]) / window_steps
            mean_i = (total_i - base[5]) / window_steps
            accumulated = (total_e + total_i - base[4] - base[5]) * self.volume / self.macro_weight   # macro-particle-steps
            net = np.abs(mean_i - mean_e)
            resolved = accumulated >= self.floor
            raw_at = int(np.argmax(net))
            far_net = float(net[resolved].max()) if resolved.any() else 0.0
            out |= {
                "far_field_net_charge_density_max_per_m3": far_net,
                "charge_fraction_of_peak": far_net / peak_mean if peak_mean > 0.0 else 0.0,
                "far_field_net_charge_density_max_window_raw_per_m3": float(net[raw_at]),
                "charge_fraction_of_peak_window_raw": float(net[raw_at]) / peak_mean if peak_mean > 0.0 else 0.0,
                "far_field_window_raw_max_node": self._node(raw_at),
                "far_field_window_raw_max_accumulated_macro_particles": float(accumulated[raw_at]),
                "far_field_resolved_nodes": int(resolved.sum()),
                "far_field_accumulated_macro_particles_max": float(accumulated.max()),
                "far_field_accumulated_macro_particles_median": float(np.median(accumulated)),
            }
        else:
            out |= {
                "far_field_net_charge_density_max_per_m3": 0.0, "charge_fraction_of_peak": 0.0,
                "far_field_net_charge_density_max_window_raw_per_m3": 0.0, "charge_fraction_of_peak_window_raw": 0.0,
                "far_field_window_raw_max_node": [0, 0], "far_field_window_raw_max_accumulated_macro_particles": 0.0,
                "far_field_resolved_nodes": 0, "far_field_accumulated_macro_particles_max": 0.0,
                "far_field_accumulated_macro_particles_median": 0.0,
            }
        self._ring.append((int(step), total_steps, self._peak_total, self._records, total_e.copy(), total_i.copy()))
        return out

    def _node(self, flat_far_index: int) -> list[int]:
        node = np.flatnonzero(self.far.ravel())[flat_far_index]
        return [int(k) for k in np.unravel_index(int(node), self.far.shape)]


def boundary_forces_n(masks: MeshMasks, phi: np.ndarray) -> dict[str, float]:
    """Electrostatic axial force on the solid boundaries from the discrete field (v2.0): Maxwell stress.

    The force on a body is the Maxwell stress integrated over the plasma-side faces of the
    plasma cell mask that touch it, ``F = sum_faces (eps0 [(E.n) E - E^2 n / 2]) . z  dA`` with
    ``n`` the normal from the body into the plasma and ``E`` the mean of the two nodal fields on
    the face.  This contains the surface-charge term ``sigma E`` (``E_n = sigma / eps0`` on a
    dielectric with zero field inside, the induced charge on a conductor) AND the pressure of
    the tangential field ``-eps0 E_t^2 / 2`` on the Neumann (insulator) faces, which a
    surface-charge-times-field formula misses: the cone stair-steps and the dielectric ring of
    the front face carry an axial component of it.  Returns the force on the thruster (anode +
    dielectric + front-face conductor) and on the far-field boundary (the chamber); Newton's
    third law ``F_plasma + F_thruster + F_far = 0`` holds to the discretisation error of the
    one-sided boundary fields (verified first order in ``tests/pic2d/test_pic2d_v20_plume.py``).
    """

    grid = masks.grid
    geometry = grid.geometry
    nr, nz = grid.cell_shape
    r = grid.r_m
    dz = grid.dz_m
    e_r, e_z = electric_field_nodes(masks, phi)
    plasma_cell = masks.plasma_cell
    # stress components at nodes: T_zz = eps0 (E_z^2 - E_r^2) / 2, T_zr = eps0 E_z E_r
    t_zz = 0.5 * EPSILON_0_F_PER_M * (e_z**2 - e_r**2)
    t_zr = EPSILON_0_F_PER_M * e_z * e_r
    ring_area = pi * (r[1:] ** 2 - r[:-1] ** 2)                     # z-facet areas per radial cell
    parts = {"anode_n": 0.0, "dielectric_n": 0.0, "body_conductor_n": 0.0, "far_field_n": 0.0}
    j_exit = int(round(geometry.channel_length_m / dz)) if geometry.has_plume else nz
    i_exit = int(round(geometry.exit_radius_m / grid.dr_m))
    i_body = int(round(float(geometry.body_dielectric_radius_m) / grid.dr_m)) if geometry.has_plume else nr
    # -- z-facets: plane z_j' between cell columns j'-1 and j' (or the domain ends)
    padded = np.zeros((nr, nz + 2), dtype=bool)
    padded[:, 1:-1] = plasma_cell
    for jp in range(nz + 1):
        below, above = padded[:, jp], padded[:, jp + 1]          # cells on the -z / +z side of the plane
        face = below ^ above
        if not face.any():
            continue
        cells = np.flatnonzero(face)
        # normal from the body into the plasma: +z when the plasma is above (body below), -z otherwise
        sign = np.where(above[cells], 1.0, -1.0)
        stress = 0.5 * (t_zz[cells, jp] + t_zz[cells + 1, jp])    # face value: mean of its two nodes
        force = sign * stress * ring_area[cells]
        for cell, value in zip(cells, force):
            if jp == 0:
                parts["anode_n"] += value
            elif jp == nz:
                parts["far_field_n"] += value
            elif geometry.has_plume and jp == j_exit and cell >= i_exit:
                parts["body_conductor_n" if cell >= i_body else "dielectric_n"] += value
            else:
                parts["dielectric_n"] += value
    # -- r-facets: cylinder r_i' between cell rows i'-1 and i' (or the outer box edge)
    padded = np.zeros((nr + 2, nz), dtype=bool)
    padded[1:-1, :] = plasma_cell
    for ip in range(1, nr + 1):
        inner, outer = padded[ip, :], padded[ip + 1, :]
        face = inner ^ outer
        if not face.any():
            continue
        cells = np.flatnonzero(face)
        # normal from the body into the plasma: -r when the plasma is inside (body outside), +r otherwise
        sign = np.where(inner[cells], -1.0, 1.0)
        stress = 0.5 * (t_zr[ip, cells] + t_zr[ip, cells + 1])
        force = sign * stress * (2.0 * pi * r[ip] * dz)
        for cell, value in zip(cells, force):
            if ip == nr and (geometry.has_plume and cell >= j_exit):
                parts["far_field_n"] += value
            else:
                parts["dielectric_n"] += value
    parts["thruster_n"] = parts["anode_n"] + parts["dielectric_n"] + parts["body_conductor_n"]
    return parts


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
        step_graph: bool = True,
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

            self.backend = WarpBackend(config, self.masks, field, cross_sections, device=("cpu" if backend == "warp-cpu" else device),
                                       step_graph=step_graph)
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
            # v2.0: the inventory is the *channel* zone; the plume density is the analytic effusion cone
            self.neutrals = NeutralInventory(
                config.neutral_inventory,
                ceiling_density_per_m3=config.mcc.neutral_density_per_m3,
                exit_area_m2=pi * geometry.exit_radius_m**2,
                temperature_k=config.mcc.neutral_temperature_k,
                volume_m3=float(self.masks.channel_volume_m3 if self.masks.has_plume else self.masks.plasma_volume_m3),
            )
            # v2.0: the inventory may start below the null-collision ceiling (declared headroom for the
            # recycling transient above Q/c); the MCC scale is n_g / ceiling from the first step
            self.neutral_state = NeutralState.initial(self.neutrals.initial_density)
            self.backend.set_neutral_scale(self.neutrals.scale(self.neutral_state))
        self._last_momentum: float | None = None
        if config.cathode is not None and config.cathode.current_rule == "continuity":
            self.backend.set_emission_rate(config.initial_emission_rate_per_step)
        # v2.0.2: trailing-window far-field charge statistic (plume domains only; recording values when the gate is off)
        self._far_field_window: FarFieldChargeWindow | None = None
        if self.masks.has_plume:
            gate = config.plume_boundary_gate
            self._far_field_window = FarFieldChargeWindow(
                self.masks, config.macro_weight,
                gate.window_steps if gate is not None else PLUME_GATE_WINDOW_STEPS, config.series_interval_steps,
                gate.min_accumulated_macro_particles_per_node if gate is not None else PLUME_GATE_MIN_ACCUMULATED_MACRO_PARTICLES_PER_NODE,
            )
            self._far_field_window.reset(self.backend.far_field_window_sums(), self.backend.step_index)

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
        self._last_cumulative = {key: float(state.cumulative.get(key, 0.0)) for key in CUMULATIVE_KEYS} | {
            key: float(value) for key, value in state.cumulative.items() if key not in CUMULATIVE_KEYS
        }
        self._last_energy = None
        self._last_electrode = None
        self._last_momentum = None
        self._series_base_step = int(state.step)
        if self._far_field_window is not None:
            # the window history is not part of the checkpoint: after a resume the gate statistic covers the
            # accumulation since the resume and is enforced once the window is complete again (disclosed)
            self._far_field_window.reset(self.backend.far_field_window_sums(), int(state.step))
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
        if "anomalous" in cumulative:     # v1.4 Bohm-scattering hook tally (macro collisions -> real collisions per second)
            currents["anomalous_collision_rate_per_s"] = (
                (cumulative["anomalous"] - self._last_cumulative.get("anomalous", 0.0)) * config.macro_weight / interval
            )
        extra = lambda key: cumulative.get(key, 0.0) - self._last_cumulative.get(key, 0.0)  # noqa: E731  (v2.0 extra ledger keys)
        if masks.has_plume:
            currents["body_face_electron_a"] = extra("body_face_electrons") * current_unit
            currents["body_face_ion_a"] = extra("body_face_ions") * current_unit
            currents["plume_ionization_rate_per_s"] = extra("ionizations_plume") * config.macro_weight / interval
            currents["cathode_emission_a"] = currents["injected_electron_a"]
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
            # v1.4: ions absorbed at the dielectric wall and the anode are recycled as
            # thermal neutrals (when the inventory has wall_recycling); exit ions are the beam.
            # v2.0: only channel ionisation drains the channel inventory (plume births consume effused
            # atoms) and front-face ions are not recycled into the channel.
            absorbed_ion_rate = (delta["wall_ions"] + delta["anode_ions"] - extra("body_face_ions")) * config.macro_weight / interval
            channel_ionization_rate = currents["ionization_rate_per_s"] - currents.get("plume_ionization_rate_per_s", 0.0)
            advance = self.neutrals.advance(self.neutral_state, channel_ionization_rate, interval, absorbed_ion_rate)
            self.neutral_state = advance.state
            self.backend.set_neutral_scale(self.neutrals.scale(self.neutral_state))
            feed = self.neutrals.config.feed_atoms_per_s
            neutral = {
                "density_per_m3": advance.state.density_per_m3,
                "fixed_point_per_m3": advance.fixed_point_per_m3,
                "scale": self.neutrals.scale(advance.state),
                "ionization_rate_per_s": advance.ionization_rate_per_s,
                "effusion_rate_per_s": advance.effusion_rate_per_s,
                "artificial_rate_per_s": advance.artificial_rate_per_s,
                "recycled_rate_per_s": advance.recycled_rate_per_s,
                "wall_ion_absorption_rate_per_s": absorbed_ion_rate,
                "effusion_coefficient_m3_per_s": advance.effusion_coefficient_m3_per_s,
                "feed_atoms_per_s": feed,
                "gross_utilisation": advance.ionization_rate_per_s / feed,
                "net_utilisation": (advance.ionization_rate_per_s - advance.recycled_rate_per_s) / feed,
                "interval_ledger_residual_atoms": advance.ledger_residual_atoms,
                "ledger": dict(advance.state.ledger),
            }
        # v1.4: peak-node Debye sample (blocker 1): the grid must resolve the PEAK, not the mean
        gate = config.peak_debye_gate
        peak_node = self.backend.peak_node_sample()
        if gate is not None:
            peak_node["gate_max_cells_per_debye"] = gate.max_cells_per_debye
            enforced = peak_node["macro_particles_at_peak"] >= gate.min_macro_particles_at_peak
            peak_node["gate_enforced"] = bool(enforced)
        momentum: dict[str, Any] | None = None
        plume: dict[str, Any] | None = None
        if masks.has_plume:
            momentum = self._momentum_record(sample, extra, interval, neutral)
            plume = self._plume_record(sample, phi, float(sample["time_s"]))
            if config.cathode is not None and config.cathode.current_rule == "continuity":
                self._continuity_update(delta, interval_steps, momentum)
        self.series.append(
            SeriesRecord(
                step, float(sample["time_s"]), int(sample["electrons"]), int(sample["ions"]),
                float(plasma_phi.mean()), float(plasma_phi.min()), float(plasma_phi.max()),
                k_e, k_i, u_e, float(sample["surface_charge_c"]), tally.max_omega_pe_dt,
                tally.poisson_iterations, currents, ledger, neutral, peak_node, momentum, plume,
            )
        )
        self._last_cumulative = dict(cumulative)
        self._last_energy = total
        if plume is not None and plume.get("gate_enforced") and plume["charge_fraction_of_peak"] > plume["gate_max_charge_fraction"]:
            raise PIC2DStabilityError(
                f"plume-boundary gate: net charge density at the far-field nodes is {plume['charge_fraction_of_peak']:.3g} of the "
                f"peak electron density (limit {plume['gate_max_charge_fraction']}): a sheath is forming on the box boundary"
            )
        if gate is not None and peak_node["gate_enforced"] and peak_node["cells_per_debye"] > gate.max_cells_per_debye:
            raise PIC2DStabilityError(
                f"peak-node Debye gate: {peak_node['cells_per_debye']:.3g} cells per lambda_D at node {peak_node['node']} "
                f"(n_e {peak_node['n_e_peak_per_m3']:.3g} m^-3, T_e {peak_node['t_e_peak_ev']:.3g} eV) exceeds {gate.max_cells_per_debye}"
            )

    # -- v2.0 plume block ---------------------------------------------------------
    def _momentum_record(self, sample: Mapping[str, Any], extra: Any, interval: float, neutral: Mapping[str, Any] | None) -> dict[str, Any]:
        """Interval momentum ledger, momentum-flux thrust and momentum-balance thrust with the closure check.

        Signs: +z is the beam direction.  ``thrust_flux_n`` is the axial momentum leaving through the
        far-field boundary minus the momentum brought in by the cathode (plus the cold-gas effusion
        thrust when an inventory exists).  ``force_on_thruster_n`` is the axial force on the thruster
        from the plasma: momentum deposited on its surfaces minus the reaction to the field impulse on
        the plasma (E and B, the magnets are part of the thruster); in steady state
        ``thrust_flux ~ -force_on_thruster`` up to the momentum handed to the neutral gas in
        collisions, the momentum of ionisation products, the far-field (chamber) electrostatic force
        and the change of the stored plasma momentum, all of which are reported.  The ledger residual
        ``dP - (impulse + collisions + born + injected - exit - wall - anode)`` is round-off.
        """

        rate = lambda key: extra(key) / interval  # noqa: E731
        p_now = float(sample["momentum_z_electrons"]) + float(sample["momentum_z_ions"])
        d_p = 0.0 if self._last_momentum is None else p_now - self._last_momentum
        first = self._last_momentum is None
        self._last_momentum = p_now
        exit_rate = rate("pz_exit_electrons") + rate("pz_exit_ions")
        absorbed_rate = rate("pz_wall_electrons") + rate("pz_wall_ions") + rate("pz_anode_electrons") + rate("pz_anode_ions")
        impulse_rate = rate("pz_impulse")
        electric_rate = rate("pz_impulse_electric")
        collisions_rate = rate("pz_collisions")
        born_rate = rate("pz_born")
        injected_rate = rate("pz_injected")
        residual = 0.0 if first else d_p - (extra("pz_impulse") + extra("pz_collisions") + extra("pz_born") + extra("pz_injected")
                                            - extra("pz_exit_electrons") - extra("pz_exit_ions") - extra("pz_wall_electrons")
                                            - extra("pz_wall_ions") - extra("pz_anode_electrons") - extra("pz_anode_ions"))
        cold_gas = 0.0
        if neutral is not None and self.neutrals is not None:
            # effusing half-Maxwellian through the aperture: momentum flux n k T / 2 per area = Phi m (pi/4) v_bar
            v_bar = sqrt(8.0 * 1.380649e-23 * self.neutrals.temperature_k / (pi * self.neutrals.mass_kg))
            cold_gas = float(neutral["effusion_rate_per_s"]) * self.neutrals.mass_kg * (pi / 4.0) * v_bar
        thrust_flux = exit_rate - injected_rate
        force_on_thruster = absorbed_rate - impulse_rate
        thrust_balance = -force_on_thruster
        forces = boundary_forces_n(self.masks, sample["phi_v"])
        closure = (thrust_flux - thrust_balance) / thrust_flux if thrust_flux != 0.0 else 0.0
        return {
            "momentum_z_kg_m_s": p_now,
            "interval_dp_kg_m_s": d_p,
            "interval_ledger_residual_kg_m_s": residual,
            "beam_momentum_rate_ions_n": rate("pz_exit_ions"),
            "beam_momentum_rate_electrons_n": rate("pz_exit_electrons"),
            "injected_momentum_rate_n": injected_rate,
            "absorbed_momentum_rate_n": absorbed_rate,
            "field_impulse_rate_n": impulse_rate,
            "electric_impulse_rate_n": electric_rate,
            "magnetic_impulse_rate_n": impulse_rate - electric_rate,
            "collision_momentum_rate_n": collisions_rate,
            "born_momentum_rate_n": born_rate,
            "dp_rate_n": d_p / interval,
            "thrust_flux_n": thrust_flux,
            "cold_gas_thrust_n": cold_gas,
            "thrust_total_n": thrust_flux + cold_gas,
            "force_on_thruster_n": force_on_thruster,
            "thrust_balance_n": thrust_balance,
            "closure_fraction": closure,
            "electrostatic_force_thruster_n": forces["thruster_n"],
            "electrostatic_force_far_field_n": forces["far_field_n"],
            "electrostatic_force_parts_n": {k: forces[k] for k in ("dielectric_n", "anode_n", "body_conductor_n")},
        }

    def _plume_record(self, sample: Mapping[str, Any], phi: np.ndarray, time_s: float) -> dict[str, Any]:
        """Far-field boundary sample: potential consistency, charge pile-up fraction, axis potential profile markers."""

        masks = self.masks
        config = self.config
        grid = masks.grid
        q_e, q_i = self.backend.charge_maps()
        gate = config.plume_boundary_gate
        with np.errstate(invalid="ignore", divide="ignore"):
            volume = np.where(masks.plasma_node, masks.shape_volume_m3, np.inf)
            n_e = np.abs(q_e) / (ELEMENTARY_CHARGE_C * volume)
            net = (q_i + q_e) / (ELEMENTARY_CHARGE_C * volume)
        peak = float(n_e[masks.plasma_node].max()) if masks.plasma_node.any() else 0.0
        far = masks.far_field_node
        # v2.0.1 single-deposit statistic, kept as the shot-noise WITNESS only (plume attempt 6 stopped on one
        # macro-ion at the axis corner node; attempt 7 showed no far-field node reaches 32 macro-particles in one
        # deposit): the unrestricted maximum over the far-field nodes, its node and macro-particle count.
        macro = (np.abs(q_e) + np.abs(q_i)) / (ELEMENTARY_CHARGE_C * config.macro_weight)
        far_abs = np.abs(net)
        far_net_raw = float(np.max(far_abs[far])) if far.any() else 0.0
        raw_node = [int(k) for k in np.unravel_index(int(np.argmax(np.where(far, far_abs, -1.0))), net.shape)] if far.any() else [0, 0]
        # v2.0.2 gate quantity: the trailing-window average from the diagnostic accumulators (see FarFieldChargeWindow)
        assert self._far_field_window is not None
        window = self._far_field_window.update(self.backend.far_field_window_sums(), int(sample["step"]), peak)
        phi_dev = float(np.max(np.abs(phi[far] - config.potentials.exit_v))) if far.any() else 0.0
        induced = apply_operator(masks, phi)
        # axis potential: exit-plane value and the acceleration region (90 % -> 10 % of the drop from
        # the axis maximum to the far plane)
        axis = phi[0, :]
        z = grid.z_m
        j_exit = int(round(grid.geometry.channel_length_m / grid.dz_m))
        k_max = int(np.argmax(axis))
        phi_max = float(axis[k_max])
        phi_far = float(axis[-1])
        drop = phi_max - phi_far
        z90 = z10 = float("nan")
        if drop > 0.0:
            tail = axis[k_max:]
            below90 = np.flatnonzero(tail <= phi_far + 0.9 * drop)
            below10 = np.flatnonzero(tail <= phi_far + 0.1 * drop)
            if below90.size:
                z90 = float(z[k_max + below90[0]])
            if below10.size:
                z10 = float(z[k_max + below10[0]])
        record = {
            "far_field_phi_max_abs_deviation_v": phi_dev,
            # v2.0.2 gate quantity and its window (interval averages over the accumulated steps of the trailing window)
            "far_field_net_charge_density_max_per_m3": window["far_field_net_charge_density_max_per_m3"],
            "charge_fraction_of_peak": window["charge_fraction_of_peak"],
            "far_field_window_steps": window["window_steps"],
            "far_field_window_records": window["window_records"],
            "far_field_window_start_step": window["window_start_step"],
            "far_field_window_complete": window["window_complete"],
            "far_field_window_steps_required": window["window_steps_required"],
            "far_field_resolved_nodes": window["far_field_resolved_nodes"],
            "min_accumulated_macro_particles_per_node": window["min_accumulated_macro_particles_per_node"],
            "far_field_accumulated_macro_particles_max": window["far_field_accumulated_macro_particles_max"],
            "far_field_accumulated_macro_particles_median": window["far_field_accumulated_macro_particles_median"],
            # unrestricted window statistic (all far-field nodes) and its node
            "far_field_net_charge_density_max_window_raw_per_m3": window["far_field_net_charge_density_max_window_raw_per_m3"],
            "charge_fraction_of_peak_window_raw": window["charge_fraction_of_peak_window_raw"],
            "far_field_window_raw_max_node": window["far_field_window_raw_max_node"],
            "far_field_window_raw_max_accumulated_macro_particles": window["far_field_window_raw_max_accumulated_macro_particles"],
            "peak_electron_density_per_m3": peak,
            "peak_electron_density_window_per_m3": window["peak_electron_density_window_per_m3"],
            # v2.0.1 single-deposit statistic (the attempt-6 gate quantity; shot-noise witness) and its node
            "far_field_net_charge_density_max_raw_per_m3": far_net_raw,
            "charge_fraction_of_peak_raw": far_net_raw / peak if peak > 0.0 else 0.0,
            "far_field_raw_max_node": raw_node,
            "far_field_raw_max_macro_particles": float(macro[raw_node[0], raw_node[1]]) if far.any() else 0.0,
            "far_field_induced_charge_c": float(induced[far].sum()) if far.any() else 0.0,
            "body_conductor_induced_charge_c": float(induced[masks.body_conductor_node].sum()),
            "exit_plane_axis_potential_v": float(axis[j_exit]),
            "axis_phi_max_v": phi_max,
            "axis_phi_max_z_m": float(z[k_max]),
            "acceleration_z90_m": z90,
            "acceleration_z10_m": z10,
            "acceleration_width_m": (z10 - z90) if isfinite(z10) and isfinite(z90) else float("nan"),
            "cathode_rate_per_step": float(sample["cumulative"].get(CATHODE_RATE_KEY, config.initial_emission_rate_per_step)),
        }
        if gate is not None:
            record["gate_max_charge_fraction"] = gate.max_charge_fraction
            # armed = past the arming time; enforced = armed AND the trailing window holds >= window_steps accumulated
            # steps (a short window is the single-deposit shot noise again, e.g. right after a resume)
            record["gate_armed"] = bool(time_s >= gate.enforce_after_s)
            record["gate_enforced"] = bool(record["gate_armed"] and window["window_complete"])
        return record

    def _continuity_update(self, delta: Mapping[str, float], interval_steps: int, momentum: dict[str, Any]) -> None:
        """Cathode current-continuity rule: emit the discharge current, relaxed and clamped.

        Charge conservation of the whole plasma in steady state gives ``I_cathode = I_d + (I_e - I_i)_far``:
        the neutraliser in series with the anode supply emits the anode current, and the plume boundary
        then carries no net current to the chamber.  The target is the interval's net anode electron
        current (electrons absorbed minus ions absorbed) in macro-particles per step.
        """

        config = self.config
        cathode = config.cathode
        assert cathode is not None
        unit = config.dt_s / (ELEMENTARY_CHARGE_C * config.macro_weight)
        floor = cathode.current_a * unit
        ceiling = float(cathode.max_current_a) * unit
        current = float(self.backend.emission_rate_per_step)
        target = (delta["anode_electrons"] - delta["anode_ions"]) / max(interval_steps, 1)
        relaxed = current + (target - current) / cathode.continuity_relaxation_intervals
        new_rate = min(max(relaxed, floor), ceiling)
        self.backend.set_emission_rate(new_rate)
        momentum["cathode_target_rate_per_step"] = target
        momentum["cathode_rate_per_step"] = new_rate
        momentum["cathode_emission_next_a"] = new_rate / unit
        momentum["cathode_clamped"] = bool(new_rate != relaxed)

    def diagnostic_arrays(self) -> dict[str, np.ndarray]:
        return self.backend.diagnostic_arrays()

    def diagnostic_sums(self) -> dict[str, np.ndarray]:
        """v2.0: cumulative window sums for the frame recorder (both backends)."""

        return self.backend.diagnostic_sums()

    def surface_charge_map(self) -> np.ndarray:
        """Instantaneous dielectric surface charge per node (C), without pulling the particles."""

        return self.backend.surface_charge_map()

    def step_graph_state(self) -> bool | str:
        """CUDA-graph state of the step: ``True`` (captured), ``"lazy"`` (enabled, captured on the first step), ``False``."""

        if getattr(self.backend, "step_graph_active", False):
            return True
        return "lazy" if getattr(self.backend, "step_graph", False) else False

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
        inventory = self.config.neutral_inventory
        record["v1_4_options"] = {
            "wall_recycling": bool(inventory is not None and inventory.wall_recycling),
            "neutral_relaxation": (
                "no inventory" if inventory is None
                else ("artificial (development only)" if inventory.relaxation_time_s is not None else "off (physical effusion time scale)")
            ),
            "peak_debye_gate": None if self.config.peak_debye_gate is None else self.config.peak_debye_gate.to_dict(),
            "anomalous": None if self.config.anomalous is None else self.config.anomalous.to_dict(),
            "see": None if self.config.see is None else self.config.see.to_dict(),
            # True once a step graph has been captured, "lazy" when graphs are enabled but none is captured yet (the
            # capture happens on the first step, so the launch-time provenance line used to read False), False when off
            "step_graph": self.step_graph_state(),
        }
        geometry = self.config.grid.geometry
        if geometry.has_plume:
            record["v2_0_options"] = {
                "domain": "channel + plume box (L-shaped plasma region; channel walls, exit lip and front face internal)",
                "plume_radius_m": geometry.plume_radius_m, "plume_length_m": geometry.plume_length_m,
                "front_face": f"dielectric to r = {geometry.body_dielectric_radius_m} m, grounded conductor beyond (reference potential)",
                "far_field": f"Dirichlet {self.config.potentials.exit_v} V on r = R_plume (z >= z_exit) and z = z_max",
                "cathode": None if self.config.cathode is None else self.config.cathode.to_dict(),
                "legacy_exit_plane_injection": self.config.injection is not None,
                "neutrals": "two-zone: channel inventory (v1.4) + free-molecular cosine-source cone in the plume (electron-neutral MCC only; ion-neutral collisions off)",
                "plume_boundary_gate": None if self.config.plume_boundary_gate is None else self.config.plume_boundary_gate.to_dict(),
                "momentum_ledger_keys": list(MOMENTUM_KEYS),
                "histograms": {"theta_bins": THETA_BINS, "iedf_bins": IEDF_BINS, "iedf_max_ev": iedf_max_ev(self.config)},
            }
        return record


__all__ = [
    "CATHODE_RATE_KEY",
    "CPUBackend",
    "CUMULATIVE_KEYS",
    "CathodeConfig",
    "DiagnosticAccumulator",
    "FarFieldChargeWindow",
    "IEDF_BINS",
    "InjectionConfig",
    "MOMENTUM_KEYS",
    "PIC2DConfig",
    "PLUME_GATE_MIN_ACCUMULATED_MACRO_PARTICLES_PER_NODE",
    "PLUME_GATE_WINDOW_STEPS",
    "PeakDebyeGateConfig",
    "PlumeBoundaryGateConfig",
    "SeedPlasmaConfig",
    "SeriesRecord",
    "Simulation",
    "SimulationState",
    "StepTally",
    "THETA_BINS",
    "boundary_forces_n",
    "cathode_sample",
    "empty_cumulative",
    "iedf_max_ev",
    "instantaneous_maps",
    "momentum_z_kg_m_s",
    "neutral_shape_cells",
    "plume_neutral_shape",
    "seed_plasma_state",
]
