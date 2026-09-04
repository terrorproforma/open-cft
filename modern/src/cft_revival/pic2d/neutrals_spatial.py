"""Spatially resolved test-particle neutrals + Xe(6s) metastables (model v2.5.0; CPU reference).

``neutrals_spatial_v1`` replaces the 0-D inventory of :mod:`cft_revival.pic2d.neutrals` (which stays
as ``neutrals.model = "inventory-0d"``, bitwise) by neutral MACRO-PARTICLES in free-molecular flight
(DSMC-lite, collisionless: at n_g 1e19-1e20 in a 4 mm bore Kn = lambda_nn / D ~ 10-100 so atom-atom
collisions are negligible and the transport is ballistic with wall reflection):

* atoms are injected at the anode face at the declared feed ``Q`` with a cosine-law (flux) Maxwellian
  at the feed temperature ``T_g`` (``mcc.neutral_temperature_k``);
* they fly in straight lines (the same meridional-frame position advance as the charged particles),
  reflect at the dielectric walls, the cone stair steps and the anode with the accommodation
  coefficient ``alpha``: diffuse (cosine law at the wall temperature ``T_w``) with probability
  ``alpha``, specular otherwise;
* they leave through the exit aperture (channel-only domains) or the far-field boundary (plume
  domains) - the ``effused`` ledger term, whose axial momentum / kinetic energy are the cold-gas and
  fast-neutral thrust;
* they are depleted by the plasma WEIGHT-CONSISTENTLY: every ionisation (``W`` atoms), every CEX
  event (``W`` atoms turned into the slow ion) and every excitation into the metastable pool
  (``b_k W`` atoms) is booked as a per-cell atom SINK by the electron / ion MCC; at the next neutral
  sub-step the ground weights of that cell are scaled down so that exactly the demanded atoms are
  removed (what a momentarily empty cell cannot deliver is carried as a per-cell DEBT and removed
  when atoms arrive: the ledger identity closes to round-off through the debt);
* wall-ion recycling (v1.4) happens AT THE IMPACT CELL: the push books ``recomb x W`` atoms per
  absorbed ion (dielectric wall, cone, anode; never the exit / far field / body face) into the ion's
  last plasma cell, spawned there as thermal atoms at ``T_w`` at the macro-weight granularity
  (per-cell carry);
* CEX fast neutrals (v2.3.0) become REAL particles: the ion MCC hands the ion's pre-event velocity
  over at the ion position with the ion's weight; their fate (exit, wall thermalisation) is the
  transport's, not a straight-line guess at the event;
* the density is deposited to the (r, z) CELL grid at every neutral sub-step (nearest cell, the
  same cell the MCC's two-zone shape used) into a device-resident per-cell array that the electron
  MCC and the ion MCC read INSTEAD of the scalar ``n_g(t) x shape`` (the CUDA-graph lesson of
  2026-09-04: a per-step host scalar is frozen at capture; the array is not), together with the
  per-cell drift / thermal speed the ion MCC samples its target atom from.

Neutrals are SUB-CYCLED: one neutral sub-step every ``substep_steps`` PIC steps.  The declared
``time_acceleration`` ``F`` (default 1 = physical) makes the neutrals advance ``F`` times further per
sub-step, with the feed, the recycling, the CEX hand-off weight and the plasma sinks all scaled by
``F``: the neutral system then relaxes ``F`` times faster toward the quasi-steady profile of the
CURRENT plasma (the spatial generalisation of the v1.3 artificial relaxation ``tau_g``; the physical
approach time V / c ~ 0.2 ms is ~100x a feasible run).  As with ``tau_g`` only the fixed point is
physical; the atom ledger is kept in NEUTRAL time and the real-time balance is recovered by dividing
the plasma-coupled terms by ``F`` (``advance`` records both).

``metastables_v1`` (optional block): the Xe(6s) metastable pool is a second neutral state
(``state = 1``) of the same particle arrays with its own, smaller macro-weight ``W_m`` (metastables
are ~1-2 % of the gas; equal weights would leave ~1 macro-metastable per cell).  Sources: the
declared fraction ``b_k`` of every excitation level ``k`` that ends in a metastable level (the
6s[3/2]_2 pool at ``pool_energy_ev``; the level energy above it is radiated).  Sinks: stepwise
ionisation ``e + Xe* -> Xe+ + 2e`` (threshold ``E_iz - E_m`` = 3.815 eV; Binary-Encounter-Bethe
cross section, Kim and Rudd 1994) and superelastic de-excitation ``e + Xe* -> e + Xe + E_m``
(detailed balance from the bound ground-state excitation table; the atom returns to the ground
pool) as two extra channels of the electron MCC against the LOCAL metastable density; wall
de-excitation (state flip on wall contact); an optional effective radiative decay rate (0 for true
metastables); exit through the aperture.  The electron energy ledger books the stepwise threshold
as an inelastic loss and the superelastic gain as a negative one, so the particle-side energy
identity still closes; the pool's stored energy is ``E_m`` per metastable atom by construction
(released at ionisation into the ion's potential energy, at the wall, by radiation or back to the
electron), tested through the metastable atom ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi, sqrt
from typing import Any

import numpy as np

from . import kernels
from .mcc import UniformSigmaTable, XenonCrossSections
from .mesh import MeshMasks
from .models import PIC2DValidationError, ParticleArrays
from .neutrals import BOLTZMANN_J_PER_K, XENON_MASS_KG, effusion_coefficient_m3_per_s, mean_thermal_speed_m_per_s

NEUTRAL_MODEL_INVENTORY_0D = "inventory-0d"
NEUTRAL_MODEL_SPATIAL_V1 = "neutrals_spatial_v1"
METASTABLE_MODEL_V1 = "metastables_v1"
STATE_GROUND = 0
STATE_METASTABLE = 1
# Xe I levels (NIST ASD): 6s[3/2]_2 metastable 8.3153 eV, 6s'[1/2]_0 metastable 9.4472 eV, ionisation 12.1298 eV
XE_6S_METASTABLE_EV = 8.315
# BEB constants (Kim and Rudd 1994, doi:10.1103/PhysRevA.50.3954): 4 pi a_0^2 R^2 with a_0 the Bohr radius, R = 13.6057 eV
BOHR_RADIUS_M = 5.29177210903e-11
RYDBERG_EV = 13.605693123
# RNG stream id of the neutral sub-step (the seed table column 6 on the Warp backend, after the v2.4.0 Coulomb column 5;
# CPU default_rng([seed, step, 7]))
NEUTRAL_RNG_STREAM = 7
# The plasma's atom sinks are booked as INTEGER counts in units of the ion macro weight / 2**20 on both backends (order-independent
# accumulation on the device), so a branching fraction or recombination coefficient enters as rint(b 2**20) / 2**20 (1e-6 resolution)
SINK_FIXED_POINT = float(2**20)


def quantised_fraction(value: float) -> float:
    """``rint(value 2**20) / 2**20``: the fraction as the sink bookkeeping of both backends applies it."""

    return float(np.rint(value * SINK_FIXED_POINT) / SINK_FIXED_POINT)

# cumulative-ledger keys (atoms, NEUTRAL time, macro weight applied) written by both backends as EXTRA keys
NEUTRAL_SPATIAL_LEDGER_KEYS = (
    "neutral_fed", "neutral_recycled", "neutral_fast_in", "neutral_ionized", "neutral_cex_converted", "neutral_excited_to_pool",
    "neutral_effused", "neutral_returned", "neutral_wall_hits", "neutral_anode_hits", "neutral_removed_ground", "neutral_removed_meta",
    "neutral_pz_exit", "neutral_ke_exit_j", "neutral_pz_wall", "neutral_ceiling_violations", "neutral_substeps",
    "meta_produced", "meta_ionized", "meta_superelastic", "meta_wall_deexcited", "meta_radiative", "meta_effused",
)


def clausing_factor(length_over_diameter: float) -> float:
    """Clausing transmission probability of a cylindrical tube (Clausing 1932; table as in O'Hanlon 2003, Table 3.3).

    Log-linear interpolation of the tabulated values; used by the conductance test, not by the model.
    """

    table_ld = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0])
    table_k = np.array([1.0, 0.6720, 0.5136, 0.4205, 0.3564, 0.2750, 0.2316, 0.1973, 0.1719, 0.1370, 0.1135, 0.0797, 0.0613])
    x = float(length_over_diameter)
    if not isfinite(x) or x < 0.0:
        raise PIC2DValidationError("length_over_diameter must be finite and >= 0")
    if x >= table_ld[-1]:
        return float(table_k[-1] * table_ld[-1] / x)     # long-tube limit K ~ 4 D / (3 L)
    return float(np.exp(np.interp(x, table_ld, np.log(table_k))))


def knudsen_profile_per_m3(geometry: Any, z_m: np.ndarray, feed_atoms_per_s: float, temperature_k: float,
                           exit_density_per_m3: float | None = None, mass_kg: float = XENON_MASS_KG) -> np.ndarray:
    """Knudsen-diffusion density profile of a closed-end tube fed at the anode: ``dn/dz = -3 Q / (2 pi a(z)^3 v_bar)``.

    ``n(z_exit)`` is the effusion density ``4 Q / (v_bar A_exit)`` (the 0-D zero-ionisation value) unless given.  Exact in
    the long-tube limit (Knudsen 1909: D_K = 2 a v_bar / 3); the end correction is the transport's to find.
    """

    z = np.asarray(z_m, dtype=np.float64)
    v_bar = mean_thermal_speed_m_per_s(temperature_k, mass_kg)
    n_exit = feed_atoms_per_s / effusion_coefficient_m3_per_s(pi * geometry.exit_radius_m**2, temperature_k, mass_kg) \
        if exit_density_per_m3 is None else float(exit_density_per_m3)
    # integrate the gradient from z to z_exit on a fine grid (the cone radius varies)
    fine = np.linspace(geometry.z_min_m, geometry.z_max_m, 4097)
    radius = np.asarray(geometry.wall_radius_m(fine), dtype=np.float64)
    if geometry.has_plume:
        radius = np.where(fine >= geometry.z_max_m, geometry.exit_radius_m, radius)
    gradient = 3.0 * feed_atoms_per_s / (2.0 * pi * radius**3 * v_bar)
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (gradient[1:] + gradient[:-1]) * np.diff(fine))))
    total = cumulative[-1]
    excess = np.interp(np.clip(z, geometry.z_min_m, geometry.z_max_m), fine, total - cumulative)
    return n_exit + np.where(z >= geometry.z_max_m, 0.0, excess)


def beb_cross_section_m2(energy_ev: np.ndarray, binding_ev: float, kinetic_ev: float, electrons: float = 1.0) -> np.ndarray:
    """Binary-Encounter-Bethe ionisation cross section (Kim and Rudd 1994, eq. 57 with the ``Q = 1`` simplification).

    ``sigma = S / (t + u + 1) [ (ln t / 2)(1 - 1/t^2) + 1 - 1/t - ln t / (t + 1) ]`` with ``t = E/B``, ``u = U/B``,
    ``S = 4 pi a_0^2 N (R/B)^2``.  Zero below the binding energy.
    """

    e = np.asarray(energy_ev, dtype=np.float64)
    t = e / binding_ev
    u = kinetic_ev / binding_ev
    s = 4.0 * pi * BOHR_RADIUS_M**2 * electrons * (RYDBERG_EV / binding_ev) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_t = np.log(np.maximum(t, 1.0))
        bracket = 0.5 * ln_t * (1.0 - 1.0 / np.maximum(t, 1.0) ** 2) + 1.0 - 1.0 / np.maximum(t, 1.0) - ln_t / (np.maximum(t, 1.0) + 1.0)
        sigma = s / (t + u + 1.0) * bracket
    return np.where(t > 1.0, np.maximum(sigma, 0.0), 0.0)


@dataclass(frozen=True, slots=True)
class MetastableConfig:
    """``metastables_v1``: branching of the excitation levels into the Xe(6s) pool and the pool's sinks (part of ``config_sha256``)."""

    # fraction of each excitation level's events (table order of the collision set) that ends in the metastable pool.
    # Defaults for the Biagi-v7.1 four-level set (8.315 / 9.447 / 9.917 / 11.7 eV): (0.45, 0.35, 0.50, 0.35) - see the
    # v2.5.0 spec entry for the derivation (6s lumps: metastable share of the (met + resonance) pair from the BSR
    # level-resolved cross sections; 6p cascade: Aymar and Coulombe 1978 branching into 1s5 / 1s3; upper levels: cascade)
    branching: tuple[float, ...]
    pool_energy_ev: float = XE_6S_METASTABLE_EV
    # macro-weight of a metastable macro-particle relative to a ground-state one (metastables are ~1-2 % of the gas)
    weight_ratio: float = 0.02
    # null-collision ceiling of the metastable density as a fraction of the ground ceiling n_g0
    ceiling_fraction: float = 0.05
    # stepwise ionisation: BEB with binding B = E_iz - E_m and orbital kinetic energy U (None = B, virial theorem), times a scale
    beb_kinetic_ev: float | None = None
    stepwise_scale: float = 1.0
    # superelastic de-excitation e + Xe* -> e + Xe + E_m by detailed balance from the ground excitation table of the
    # level that feeds the pool (index in table order), with statistical weights g_ground / g_meta (1 / 5 for 6s[3/2]_2)
    # and the metastable share of that (lumped) level = its branching entry
    superelastic: bool = True
    superelastic_level: int = 0
    superelastic_weight_ratio: float = 0.2
    wall_deexcitation_probability: float = 1.0
    radiative_decay_rate_per_s: float = 0.0

    def __post_init__(self) -> None:
        branching = tuple(float(b) for b in self.branching)
        if not branching or any(not isfinite(b) or not 0.0 <= b <= 1.0 for b in branching):
            raise PIC2DValidationError("metastable branching must be one fraction in [0, 1] per excitation level")
        object.__setattr__(self, "branching", branching)
        if not isfinite(self.pool_energy_ev) or self.pool_energy_ev <= 0.0:
            raise PIC2DValidationError("pool_energy_ev must be positive")
        if not isfinite(self.weight_ratio) or not 0.0 < self.weight_ratio <= 1.0:
            raise PIC2DValidationError("weight_ratio must lie in (0, 1]")
        if not isfinite(self.ceiling_fraction) or not 0.0 < self.ceiling_fraction <= 1.0:
            raise PIC2DValidationError("ceiling_fraction must lie in (0, 1]")
        if self.beb_kinetic_ev is not None and (not isfinite(self.beb_kinetic_ev) or self.beb_kinetic_ev <= 0.0):
            raise PIC2DValidationError("beb_kinetic_ev must be positive or None")
        if not isfinite(self.stepwise_scale) or self.stepwise_scale <= 0.0:
            raise PIC2DValidationError("stepwise_scale must be positive")
        if not isinstance(self.superelastic, bool):
            raise PIC2DValidationError("superelastic must be a bool")
        if not isinstance(self.superelastic_level, int) or not 0 <= self.superelastic_level < len(branching):
            raise PIC2DValidationError("superelastic_level must index an excitation level")
        if not isfinite(self.superelastic_weight_ratio) or self.superelastic_weight_ratio <= 0.0:
            raise PIC2DValidationError("superelastic_weight_ratio must be positive")
        if not isfinite(self.wall_deexcitation_probability) or not 0.0 <= self.wall_deexcitation_probability <= 1.0:
            raise PIC2DValidationError("wall_deexcitation_probability must lie in [0, 1]")
        if not isfinite(self.radiative_decay_rate_per_s) or self.radiative_decay_rate_per_s < 0.0:
            raise PIC2DValidationError("radiative_decay_rate_per_s must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": METASTABLE_MODEL_V1,
            "branching": list(self.branching),
            "pool_energy_ev": self.pool_energy_ev,
            "weight_ratio": self.weight_ratio,
            "ceiling_fraction": self.ceiling_fraction,
            "beb_kinetic_ev": self.beb_kinetic_ev,
            "stepwise_scale": self.stepwise_scale,
            "superelastic": self.superelastic,
            "superelastic_level": self.superelastic_level,
            "superelastic_weight_ratio": self.superelastic_weight_ratio,
            "wall_deexcitation_probability": self.wall_deexcitation_probability,
            "radiative_decay_rate_per_s": self.radiative_decay_rate_per_s,
        }


@dataclass(frozen=True, slots=True)
class MetastableProcessTable:
    """The two electron-impact channels on the metastable pool, resampled on the electron MCC's uniform grid.

    Row 0: stepwise ionisation (threshold ``E_iz - E_m``); row 1: superelastic de-excitation (threshold 0; the electron
    GAINS ``E_m``).  ``nu_max`` at the metastable ceiling ``n_m0`` adds to the ground null-collision ceiling.
    """

    table: UniformSigmaTable
    stepwise_threshold_ev: float
    pool_energy_ev: float
    ceiling_density_per_m3: float
    superelastic_on: bool

    @classmethod
    def build(cls, cross_sections: XenonCrossSections, config: MetastableConfig, *, ground_ceiling_per_m3: float,
              energy_step_ev: float, energy_max_ev: float) -> "MetastableProcessTable":
        levels = cross_sections.excitation_levels
        if len(config.branching) != len(levels):
            raise PIC2DValidationError(f"metastable branching has {len(config.branching)} entries for {len(levels)} excitation levels")
        e_iz = float(cross_sections.processes[-1].threshold_ev)
        binding = e_iz - config.pool_energy_ev
        if binding <= 0.0:
            raise PIC2DValidationError("pool_energy_ev must lie below the ionisation threshold")
        count = int(round(energy_max_ev / energy_step_ev)) + 1
        grid = np.arange(count, dtype=np.float64) * energy_step_ev
        kinetic = binding if config.beb_kinetic_ev is None else config.beb_kinetic_ev
        stepwise = config.stepwise_scale * beb_cross_section_m2(grid, binding, kinetic)
        superelastic = np.zeros_like(grid)
        if config.superelastic:
            # Klein-Rosseland detailed balance: g_m E' sigma_super(E') = g_0 E sigma_exc(E), E = E' + E_m; the lumped level's
            # metastable share is its branching entry.  E' = 0 (the first grid point) carries the E' -> 0 limit of E' sigma.
            level = levels[config.superelastic_level]
            share = config.branching[config.superelastic_level]
            upper = grid + config.pool_energy_ev
            sigma_exc = level.at(upper)
            e_prime = np.maximum(grid, energy_step_ev)
            superelastic = config.superelastic_weight_ratio * share * (upper / e_prime) * sigma_exc
        table = UniformSigmaTable(float(energy_step_ev), float(grid[-1]), np.stack([stepwise, superelastic]), (binding, 0.0))
        return cls(table, binding, config.pool_energy_ev, config.ceiling_fraction * ground_ceiling_per_m3, config.superelastic)

    def maximum_collision_frequency(self) -> float:
        from .mcc import electron_speed_from_energy

        energy = np.arange(self.table.point_count, dtype=np.float64) * self.table.energy_step_ev
        speed = electron_speed_from_energy(energy)
        return float(self.ceiling_density_per_m3 * np.max(self.table.table_m2.sum(axis=0) * speed))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stepwise_threshold_ev": self.stepwise_threshold_ev,
            "pool_energy_ev": self.pool_energy_ev,
            "ceiling_density_per_m3": self.ceiling_density_per_m3,
            "stepwise_peak_m2": float(self.table.table_m2[0].max()),
            "stepwise_peak_energy_ev": float(self.table.energy_step_ev * int(np.argmax(self.table.table_m2[0]))),
            "superelastic_on": self.superelastic_on,
            "table_sha256": self.table.sha256(),
            "sources": [
                "stepwise ionisation: Binary-Encounter-Bethe (Y.-K. Kim and M. E. Rudd, Phys. Rev. A 50, 3954 (1994), doi:10.1103/PhysRevA.50.3954) "
                "with B = E_iz - E_m, U = B, N = 1; bracket of the literature tables: D. Ton-That and M. R. Flannery, Phys. Rev. A 15, 517 (1977), "
                "doi:10.1103/PhysRevA.15.517; H. A. Hyman, Phys. Rev. A 20, 855 (1979), doi:10.1103/PhysRevA.20.855; H. Deutsch et al., "
                "J. Phys. B 32, 4249 (1999), doi:10.1088/0953-4075/32/17/309",
                "superelastic: Klein-Rosseland detailed balance from the bound ground-state excitation table (LXCat Biagi-v7.1)",
            ],
        }


@dataclass(frozen=True, slots=True)
class SpatialNeutralConfig:
    """``neutrals_spatial_v1`` block of ``PIC2DConfig`` (every field enters ``config_sha256``)."""

    feed_atoms_per_s: float
    macro_weight: float
    substep_steps: int
    time_acceleration: float = 1.0
    wall_temperature_k: float | None = None
    accommodation_coefficient: float = 1.0
    wall_recycling: bool = True
    recombination_coefficient: float = 1.0
    initial_profile: str = "knudsen"
    initial_density_per_m3: float | None = None
    # the published density is CLAMPED at the null-collision ceiling; a clamped cell-substep is a "ceiling violation" (recorded).
    # Shot noise of the nearest-cell deposit (~1/sqrt(N_cell)) makes rare excursions unavoidable, so the run fails closed only
    # when the violating cell-substeps exceed this fraction of the plasma cell-substeps of a series interval (0 = any violation)
    max_ceiling_violation_fraction: float = 1.0e-3
    metastables: MetastableConfig | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.feed_atoms_per_s) or self.feed_atoms_per_s <= 0.0:
            raise PIC2DValidationError("neutral feed must be positive")
        if not isfinite(self.max_ceiling_violation_fraction) or not 0.0 <= self.max_ceiling_violation_fraction <= 1.0:
            raise PIC2DValidationError("max_ceiling_violation_fraction must lie in [0, 1]")
        if not isfinite(self.macro_weight) or self.macro_weight <= 0.0:
            raise PIC2DValidationError("neutral macro_weight must be positive")
        if isinstance(self.substep_steps, bool) or not isinstance(self.substep_steps, int) or self.substep_steps < 1:
            raise PIC2DValidationError("substep_steps must be a positive integer")
        if not isfinite(self.time_acceleration) or self.time_acceleration < 1.0:
            raise PIC2DValidationError("time_acceleration must be >= 1 (1 = physical neutral time)")
        if self.wall_temperature_k is not None and (not isfinite(self.wall_temperature_k) or self.wall_temperature_k <= 0.0):
            raise PIC2DValidationError("wall_temperature_k must be positive or None (= feed temperature)")
        if not isfinite(self.accommodation_coefficient) or not 0.0 <= self.accommodation_coefficient <= 1.0:
            raise PIC2DValidationError("accommodation_coefficient must lie in [0, 1]")
        if not isinstance(self.wall_recycling, bool):
            raise PIC2DValidationError("wall_recycling must be a bool")
        if not isfinite(self.recombination_coefficient) or not 0.0 <= self.recombination_coefficient <= 1.0:
            raise PIC2DValidationError("recombination_coefficient must lie in [0, 1]")
        if self.initial_profile not in ("knudsen", "uniform", "empty"):
            raise PIC2DValidationError("initial_profile must be 'knudsen', 'uniform' or 'empty'")
        if self.initial_density_per_m3 is not None and (not isfinite(self.initial_density_per_m3) or self.initial_density_per_m3 <= 0.0):
            raise PIC2DValidationError("initial_density_per_m3 must be positive or None (= Q / c)")
        if self.metastables is not None and not isinstance(self.metastables, MetastableConfig):
            raise PIC2DValidationError("metastables must be a MetastableConfig")

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "model": NEUTRAL_MODEL_SPATIAL_V1,
            "feed_atoms_per_s": self.feed_atoms_per_s,
            "macro_weight": self.macro_weight,
            "substep_steps": self.substep_steps,
            "time_acceleration": self.time_acceleration,
            "wall_temperature_k": self.wall_temperature_k,
            "accommodation_coefficient": self.accommodation_coefficient,
            "wall_recycling": self.wall_recycling,
            "recombination_coefficient": self.recombination_coefficient,
            "initial_profile": self.initial_profile,
            "initial_density_per_m3": self.initial_density_per_m3,
            "max_ceiling_violation_fraction": self.max_ceiling_violation_fraction,
        }
        if self.metastables is not None:
            record["metastables"] = self.metastables.to_dict()
        return record


@dataclass(slots=True)
class NeutralParticles:
    """SoA neutral macro-particles: position, velocity, weight (atoms) and state (0 ground, 1 metastable)."""

    r_m: np.ndarray
    z_m: np.ndarray
    vr_m_per_s: np.ndarray
    vt_m_per_s: np.ndarray
    vz_m_per_s: np.ndarray
    weight: np.ndarray
    state: np.ndarray

    @classmethod
    def empty(cls) -> "NeutralParticles":
        z = np.zeros(0, dtype=np.float64)
        return cls(z, z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), np.zeros(0, dtype=np.int32))

    @property
    def count(self) -> int:
        return int(self.r_m.size)

    def copy(self) -> "NeutralParticles":
        return NeutralParticles(*(np.array(a, copy=True) for a in self.arrays()))

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.r_m, self.z_m, self.vr_m_per_s, self.vt_m_per_s, self.vz_m_per_s, self.weight, self.state)

    def select(self, mask: np.ndarray) -> "NeutralParticles":
        return NeutralParticles(*(a[mask] for a in self.arrays()))

    def append(self, other: "NeutralParticles") -> "NeutralParticles":
        if other.count == 0:
            return self
        return NeutralParticles(*(np.concatenate((a, b)) for a, b in zip(self.arrays(), other.arrays(), strict=True)))

    def atoms(self, state: int) -> float:
        return float(np.sum(self.weight[self.state == state])) if self.count else 0.0


@dataclass(slots=True)
class SpatialNeutralState:
    """Checkpointable neutral state: particles + per-cell carries / debts + the published density fields."""

    particles: NeutralParticles
    debt_ground: np.ndarray        # (n_cells,) atoms the plasma consumed that the cell could not yet deliver
    debt_meta: np.ndarray          # metastable atoms owed to stepwise ionisation
    debt_meta_super: np.ndarray    # metastable atoms owed to superelastic de-excitation (returned to ground when paid)
    pending_feed: np.ndarray       # (n_cells,) atoms waiting to be spawned at the macro-weight granularity
    pending_recycle: np.ndarray
    pending_return: np.ndarray
    pending_meta: np.ndarray
    density_per_m3: np.ndarray     # (n_cells,) last published ground density (clamped at the ceiling)
    meta_density_per_m3: np.ndarray
    drift_r: np.ndarray            # (n_cells,) mean ground velocity components and thermal speed of the last deposit
    drift_t: np.ndarray
    drift_z: np.ndarray
    thermal_speed: np.ndarray
    neutral_time_s: float = 0.0
    substeps: int = 0

    def copy(self) -> "SpatialNeutralState":
        return SpatialNeutralState(self.particles.copy(), *(a.copy() for a in self.cell_arrays()), self.neutral_time_s, self.substeps)

    def cell_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.debt_ground, self.debt_meta, self.debt_meta_super, self.pending_feed, self.pending_recycle, self.pending_return,
                self.pending_meta, self.density_per_m3, self.meta_density_per_m3, self.drift_r, self.drift_t, self.drift_z, self.thermal_speed)

    CELL_ARRAY_KEYS = ("debt_ground", "debt_meta", "debt_meta_super", "pending_feed", "pending_recycle", "pending_return", "pending_meta",
                       "density_per_m3", "meta_density_per_m3", "drift_r", "drift_t", "drift_z", "thermal_speed")

    def true_ground_atoms(self) -> float:
        """Ground atoms including the un-spawned carries and minus the un-removed debt (the ledger's conserved quantity)."""

        return self.particles.atoms(STATE_GROUND) + float(self.pending_feed.sum() + self.pending_recycle.sum() + self.pending_return.sum()) \
            - float(self.debt_ground.sum())

    def true_meta_atoms(self) -> float:
        return self.particles.atoms(STATE_METASTABLE) + float(self.pending_meta.sum()) - float(self.debt_meta.sum() + self.debt_meta_super.sum())


@dataclass(slots=True)
class CellSinks:
    """Per-cell atom demands of one neutral sub-step interval (REAL atoms, macro weight applied, before ``F``)."""

    ground_ionization: np.ndarray
    ground_cex: np.ndarray
    ground_excitation: np.ndarray    # into the metastable pool (b_k-weighted)
    meta_ionization: np.ndarray
    meta_superelastic: np.ndarray
    recycle: np.ndarray              # recomb x W per absorbed ion, at the ion's last plasma cell
    fast_neutrals: ParticleArrays    # CEX fast neutrals handed over (the ion's velocity at the ion position)
    fast_weight: float               # atoms per handed-over particle (the ion macro weight)

    @classmethod
    def zeros(cls, n_cells: int, fast_weight: float) -> "CellSinks":
        return cls(*(np.zeros(n_cells) for _ in range(6)), ParticleArrays.empty(), float(fast_weight))

    def reset(self) -> None:
        for array in (self.ground_ionization, self.ground_cex, self.ground_excitation, self.meta_ionization, self.meta_superelastic, self.recycle):
            array.fill(0.0)
        self.fast_neutrals = ParticleArrays.empty()


@dataclass(frozen=True, slots=True)
class SubstepTally:
    """Atoms (neutral time) and momenta of one sub-step, in the order of ``NEUTRAL_SPATIAL_LEDGER_KEYS``."""

    values: dict[str, float]


def flux_maxwellian(rng_u: np.ndarray, thermal: float | np.ndarray) -> np.ndarray:
    """Normal component of a cosine-law (flux) Maxwellian, ``v_n = v_th sqrt(-2 ln u)`` (Box-Muller radius)."""

    return thermal * np.sqrt(-2.0 * np.log(np.maximum(rng_u, 1e-300)))


def gaussian_pair(u1: np.ndarray, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.sqrt(-2.0 * np.log(np.maximum(u1, 1e-300)))
    return radius * np.cos(2.0 * pi * u2), radius * np.sin(2.0 * pi * u2)


class SpatialNeutrals:
    """CPU reference operator of ``neutrals_spatial_v1`` (+ ``metastables_v1``)."""

    def __init__(self, config: SpatialNeutralConfig, masks: MeshMasks, *, temperature_k: float, ceiling_density_per_m3: float,
                 dt_s: float, ion_macro_weight: float, mass_kg: float = XENON_MASS_KG) -> None:
        self.config = config
        self.masks = masks
        self.grid = masks.grid
        self.geometry = masks.grid.geometry
        self.temperature_k = float(temperature_k)
        self.wall_temperature_k = float(config.wall_temperature_k) if config.wall_temperature_k is not None else self.temperature_k
        self.ceiling = float(ceiling_density_per_m3)
        self.mass_kg = float(mass_kg)
        self.dt_s = float(dt_s)
        self.ion_macro_weight = float(ion_macro_weight)
        self.substep_dt_s = config.substep_steps * dt_s * config.time_acceleration      # neutral time per sub-step
        self.thermal_speed = sqrt(BOLTZMANN_J_PER_K * self.temperature_k / self.mass_kg)
        self.wall_thermal_speed = sqrt(BOLTZMANN_J_PER_K * self.wall_temperature_k / self.mass_kg)
        nr, nz = self.grid.cell_shape
        self.nr, self.nz = nr, nz
        self.n_cells = nr * nz
        r = self.grid.r_m
        cell_volume = pi * (r[1:] ** 2 - r[:-1] ** 2)[:, None] * self.grid.dz_m * np.ones((1, nz))
        self.cell_volume = np.where(masks.plasma_cell, cell_volume, 0.0).ravel()
        # anode-face feed cells (the bore cells of the first column), share by area
        bore = np.flatnonzero(masks.plasma_cell[:, 0])
        share = np.zeros((nr, nz))
        share[bore, 0] = (r[bore + 1] ** 2 - r[bore] ** 2) / r[bore.max() + 1] ** 2
        self.feed_share = share.ravel()
        self.metastable_weight = config.macro_weight * (config.metastables.weight_ratio if config.metastables is not None else 1.0)
        self.march_step_m = 0.5 * min(self.grid.dr_m, self.grid.dz_m)
        self.exit_density = config.feed_atoms_per_s / effusion_coefficient_m3_per_s(pi * self.geometry.exit_radius_m**2, self.temperature_k, mass_kg)

    # ------------------------------------------------------------------ initial state
    def initial_density_profile(self) -> np.ndarray:
        """Cell-centred initial ground density (m^-3) of the declared profile."""

        nr, nz = self.nr, self.nz
        z_mid = self.geometry.z_min_m + (np.arange(nz) + 0.5) * self.grid.dz_m
        n_exit = self.exit_density if self.config.initial_density_per_m3 is None else self.config.initial_density_per_m3
        if self.config.initial_profile == "empty":
            profile = np.zeros(nz)
        elif self.config.initial_profile == "uniform":
            profile = np.full(nz, n_exit)
        else:
            profile = knudsen_profile_per_m3(self.geometry, z_mid, self.config.feed_atoms_per_s, self.temperature_k, exit_density_per_m3=n_exit, mass_kg=self.mass_kg)
        density = np.where(self.masks.plasma_cell, profile[None, :], 0.0)
        if self.geometry.has_plume:
            from .simulation import plume_neutral_shape

            r_mid = (np.arange(nr) + 0.5) * self.grid.dr_m
            shape = plume_neutral_shape(self.grid, r_mid[:, None], z_mid[None, :])
            density = np.where(self.masks.plume_cell, n_exit * shape, density)
        if np.any(density > self.ceiling * (1.0 + 1e-9)):
            raise PIC2DValidationError(
                f"initial neutral density {density.max():.4g} exceeds the null-collision ceiling {self.ceiling:.4g} m^-3 "
                "(the Knudsen anode density is ~(1 + 3 L / 8 a) x the exit density: raise mcc.neutral_density_per_m3)"
            )
        return density

    def initial_state(self, rng: np.random.Generator) -> SpatialNeutralState:
        density = self.initial_density_profile().ravel()
        expected = density * self.cell_volume / self.config.macro_weight
        counts = np.floor(expected).astype(np.int64)
        counts += (rng.random(self.n_cells) < (expected - counts)).astype(np.int64)
        cells = np.repeat(np.arange(self.n_cells), counts)
        particles = self._sample_in_cells(cells, rng, self.thermal_speed, np.full(cells.size, self.config.macro_weight), STATE_GROUND)
        state = SpatialNeutralState(particles, *(np.zeros(self.n_cells) for _ in range(len(SpatialNeutralState.CELL_ARRAY_KEYS))))
        self._deposit(state)
        return state

    def _sample_in_cells(self, cells: np.ndarray, rng: np.random.Generator, thermal: float | np.ndarray, weight: np.ndarray, state: int,
                         drift: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> NeutralParticles:
        """Uniform-in-volume positions inside the cells, isotropic Maxwellian velocities (optionally drifting)."""

        n = int(cells.size)
        if n == 0:
            return NeutralParticles.empty()
        i = cells // self.nz
        j = cells % self.nz
        u = rng.random((6, n))
        r_lo = i * self.grid.dr_m
        r_hi = r_lo + self.grid.dr_m
        r = np.sqrt(r_lo**2 + u[0] * (r_hi**2 - r_lo**2)) * (1.0 - 1e-12)
        z = self.geometry.z_min_m + (j + u[1]) * self.grid.dz_m
        g1, g2 = gaussian_pair(u[2], u[3])
        g3, _ = gaussian_pair(u[4], u[5])
        vr, vt, vz = thermal * g1, thermal * g2, thermal * g3
        if drift is not None:
            vr, vt, vz = vr + drift[0], vt + drift[1], vz + drift[2]
        return NeutralParticles(r, z, vr, vt, vz, np.asarray(weight, dtype=np.float64), np.full(n, state, dtype=np.int32))

    # ------------------------------------------------------------------ one sub-step
    def substep(self, state: SpatialNeutralState, sinks: CellSinks, rng: np.random.Generator, accumulate: bool = False,
                diagnostics: Any | None = None) -> SubstepTally:
        """Deplete -> spawn -> march -> deposit; returns the sub-step ledger (neutral time) and zeroes the sinks."""

        config = self.config
        f_acc = config.time_acceleration
        tally = {key: 0.0 for key in NEUTRAL_SPATIAL_LEDGER_KEYS}
        particles = state.particles
        cells = self._cells(particles)
        ground = particles.state == STATE_GROUND
        # 1) depletion: demanded atoms (x F) + old debt against the cell's weight; exact removal, remainder carried
        sum_g = np.bincount(cells[ground], weights=particles.weight[ground], minlength=self.n_cells)
        sum_m = np.bincount(cells[~ground], weights=particles.weight[~ground], minlength=self.n_cells)
        demand_g = f_acc * (sinks.ground_ionization + sinks.ground_cex + sinks.ground_excitation) + state.debt_ground
        removed_g = np.minimum(demand_g, sum_g)
        state.debt_ground = demand_g - removed_g
        with np.errstate(invalid="ignore", divide="ignore"):
            factor_g = np.where(sum_g > 0.0, 1.0 - removed_g / sum_g, 1.0)
        demand_iz = f_acc * sinks.meta_ionization + state.debt_meta
        demand_super = f_acc * sinks.meta_superelastic + state.debt_meta_super
        demand_m = demand_iz + demand_super
        removed_m = np.minimum(demand_m, sum_m)
        with np.errstate(invalid="ignore", divide="ignore"):
            factor_m = np.where(sum_m > 0.0, 1.0 - removed_m / sum_m, 1.0)
            share_super = np.where(demand_m > 0.0, demand_super / np.where(demand_m > 0.0, demand_m, 1.0), 0.0)
        removed_super = removed_m * share_super
        removed_iz = removed_m - removed_super
        state.debt_meta = demand_iz - removed_iz
        state.debt_meta_super = demand_super - removed_super
        factor = np.where(ground, factor_g[cells], factor_m[cells])
        particles.weight = particles.weight * factor
        tally["neutral_ionized"] += f_acc * float(sinks.ground_ionization.sum())
        tally["neutral_cex_converted"] += f_acc * float(sinks.ground_cex.sum())
        tally["neutral_excited_to_pool"] += f_acc * float(sinks.ground_excitation.sum())
        tally["meta_ionized"] += f_acc * float(sinks.meta_ionization.sum())
        tally["meta_superelastic"] += f_acc * float(sinks.meta_superelastic.sum())
        tally["neutral_removed_ground"] += float(removed_g.sum())
        tally["neutral_removed_meta"] += float(removed_m.sum())
        # 2) sources (ledger = the demanded atoms; the un-spawned remainders are per-cell carries inside the true count):
        #    feed, recycling, superelastic return (the metastable atoms actually removed for it), metastable production
        feed_atoms = config.feed_atoms_per_s * self.substep_dt_s
        state.pending_feed += feed_atoms * self.feed_share
        state.pending_recycle += f_acc * sinks.recycle
        state.pending_return += removed_super
        state.pending_meta += f_acc * sinks.ground_excitation
        tally["neutral_fed"] += feed_atoms
        tally["neutral_recycled"] += f_acc * float(sinks.recycle.sum())
        tally["neutral_returned"] += float(removed_super.sum())
        tally["meta_produced"] += f_acc * float(sinks.ground_excitation.sum())
        w_n, w_m = config.macro_weight, self.metastable_weight
        n_feed = np.floor(state.pending_feed / w_n).astype(np.int64)
        n_rec = np.floor(state.pending_recycle / w_n).astype(np.int64)
        n_ret = np.floor(state.pending_return / w_n).astype(np.int64)
        n_meta = np.floor(state.pending_meta / w_m).astype(np.int64) if config.metastables is not None else np.zeros(self.n_cells, dtype=np.int64)
        state.pending_feed -= n_feed * w_n
        state.pending_recycle -= n_rec * w_n
        state.pending_return -= n_ret * w_n
        state.pending_meta -= n_meta * w_m
        spawned = self._spawn_feed(np.repeat(np.arange(self.n_cells), n_feed), rng)
        spawned = spawned.append(self._sample_in_cells(np.repeat(np.arange(self.n_cells), n_rec), rng, self.wall_thermal_speed,
                                                        np.full(int(n_rec.sum()), w_n), STATE_GROUND))
        spawned = spawned.append(self._sample_in_cells(np.repeat(np.arange(self.n_cells), n_ret), rng, self.thermal_speed,
                                                        np.full(int(n_ret.sum()), w_n), STATE_GROUND))
        if config.metastables is not None and n_meta.sum():
            meta_cells = np.repeat(np.arange(self.n_cells), n_meta)
            thermal = np.where(state.thermal_speed[meta_cells] > 0.0, state.thermal_speed[meta_cells], self.thermal_speed)
            drift = (state.drift_r[meta_cells], state.drift_t[meta_cells], state.drift_z[meta_cells])
            spawned = spawned.append(self._sample_in_cells(meta_cells, rng, thermal, np.full(meta_cells.size, w_m), STATE_METASTABLE, drift))
        fast = sinks.fast_neutrals
        if fast.count:
            weight = np.full(fast.count, f_acc * sinks.fast_weight)
            spawned = spawned.append(NeutralParticles(fast.r_m.copy(), fast.z_m.copy(), fast.vr_m_per_s.copy(), fast.vt_m_per_s.copy(),
                                                      fast.vz_m_per_s.copy(), weight, np.zeros(fast.count, dtype=np.int32)))
            tally["neutral_fast_in"] += float(weight.sum())
        particles = particles.append(spawned)
        # 3) free flight with wall reflection, exit, metastable de-excitation
        particles = self._march(particles, rng, tally)
        state.particles = particles
        # 4) deposit: published density / moments (read by the MCC until the next sub-step)
        violations = self._deposit(state)
        tally["neutral_ceiling_violations"] += float(violations)
        tally["neutral_substeps"] += 1.0
        state.neutral_time_s += self.substep_dt_s
        state.substeps += 1
        if accumulate and diagnostics is not None:
            diagnostics.neutral_density += state.density_per_m3.reshape(self.nr, self.nz)
            diagnostics.metastable_density += state.meta_density_per_m3.reshape(self.nr, self.nz)
            diagnostics.neutral_samples += 1
        sinks.reset()
        return SubstepTally(tally)

    def _cells(self, particles: NeutralParticles) -> np.ndarray:
        i = np.clip(np.floor(particles.r_m / self.grid.dr_m).astype(np.int64), 0, self.nr - 1)
        j = np.clip(np.floor((particles.z_m - self.geometry.z_min_m) / self.grid.dz_m).astype(np.int64), 0, self.nz - 1)
        return i * self.nz + j

    def _spawn_feed(self, cells: np.ndarray, rng: np.random.Generator) -> NeutralParticles:
        """Feed atoms at the anode face: r uniform in the cell's area, z just inside the anode, cosine-law +z at T_g."""

        n = int(cells.size)
        if n == 0:
            return NeutralParticles.empty()
        i = cells // self.nz
        u = rng.random((5, n))
        r_lo = i * self.grid.dr_m
        r = np.sqrt(r_lo**2 + u[0] * ((r_lo + self.grid.dr_m) ** 2 - r_lo**2)) * (1.0 - 1e-12)
        z = np.full(n, self.geometry.z_min_m + 1e-6 * self.grid.dz_m)
        g1, g2 = gaussian_pair(u[1], u[2])
        vz = flux_maxwellian(u[3], self.thermal_speed)
        return NeutralParticles(r, z, self.thermal_speed * g1, self.thermal_speed * g2, vz, np.full(n, self.config.macro_weight),
                                np.zeros(n, dtype=np.int32))

    def _march(self, particles: NeutralParticles, rng: np.random.Generator, tally: dict[str, float]) -> NeutralParticles:
        """Straight flight for the sub-step in path-length pieces <= half a cell; walls reflect, the exit removes."""

        n = particles.count
        if n == 0:
            return particles
        geometry, grid, masks = self.geometry, self.grid, self.masks
        r, z = particles.r_m.copy(), particles.z_m.copy()
        vr, vt, vz = particles.vr_m_per_s.copy(), particles.vt_m_per_s.copy(), particles.vz_m_per_s.copy()
        w, st = particles.weight.copy(), particles.state.copy()
        remaining = np.full(n, self.substep_dt_s)
        alive = np.ones(n, dtype=bool)
        mass = self.mass_kg
        decay = self.config.metastables.radiative_decay_rate_per_s if self.config.metastables is not None else 0.0
        if decay > 0.0:
            meta = st == STATE_METASTABLE
            flip = meta & (rng.random(n) < 1.0 - np.exp(-decay * self.substep_dt_s))
            tally["meta_radiative"] += float(w[flip].sum())
            st[flip] = STATE_GROUND
        active = np.flatnonzero(alive)
        for _ in range(64 * (self.nr + self.nz) + 64):
            if active.size == 0:
                break
            speed = np.sqrt(vr[active] ** 2 + vt[active] ** 2 + vz[active] ** 2)
            with np.errstate(divide="ignore"):
                piece = np.where(speed > 0.0, self.march_step_m / speed, np.inf)
            dt = np.minimum(remaining[active], piece)
            r_new, z_new, vr_new, vt_new, _, _ = kernels.advance_positions(r[active], z[active], vr[active], vt[active], vz[active], dt)
            i0 = np.clip(np.floor(r[active] / grid.dr_m).astype(np.int64), 0, self.nr - 1)
            j0 = np.clip(np.floor((z[active] - geometry.z_min_m) / grid.dz_m).astype(np.int64), 0, self.nz - 1)
            # exit: through the exit plane inside the aperture (channel) / the far-field boundary (plume)
            exited = z_new >= geometry.domain_z_max_m
            if geometry.has_plume:
                exited |= (z_new >= geometry.z_max_m) & (r_new >= geometry.max_radius_m)
            else:
                exited &= r_new < geometry.exit_radius_m
            anode = z_new < geometry.z_min_m
            i1 = np.floor(r_new / grid.dr_m).astype(np.int64)
            j1 = np.clip(np.floor((z_new - geometry.z_min_m) / grid.dz_m).astype(np.int64), 0, self.nz - 1)
            beyond = i1 >= self.nr
            inside = ~beyond & ~anode & ~exited
            inside[inside] = masks.plasma_cell[np.clip(i1[inside], 0, self.nr - 1), j1[inside]]
            wall = ~inside & ~exited
            idx = active
            # exits
            ex = idx[exited]
            if ex.size:
                alive[ex] = False
                ground_ex = st[ex] == STATE_GROUND
                tally["neutral_effused"] += float(w[ex][ground_ex].sum())
                tally["meta_effused"] += float(w[ex][~ground_ex].sum())
                tally["neutral_pz_exit"] += float(np.sum(w[ex] * mass * vz[ex]))
                tally["neutral_ke_exit_j"] += float(np.sum(0.5 * mass * w[ex] * (vr_new[exited] ** 2 + vt_new[exited] ** 2 + vz[ex] ** 2)))
            # free movers
            mv = inside
            r[idx[mv]], z[idx[mv]] = r_new[mv], z_new[mv]
            vr[idx[mv]], vt[idx[mv]] = vr_new[mv], vt_new[mv]
            remaining[idx[mv]] -= dt[mv]
            # wall / anode reflections at the pre-move position (still inside the plasma)
            wl = idx[wall]
            if wl.size:
                k = np.flatnonzero(wall)
                normal_r = np.zeros(wl.size)
                normal_z = np.zeros(wl.size)
                is_anode = anode[k]
                radial = ~is_anode & ((i1[k] != i0[k]) | (beyond[k]))
                axial = ~is_anode & ~radial
                normal_z[is_anode] = 1.0
                normal_r[radial] = np.where(i1[k][radial] > i0[k][radial], -1.0, 1.0)
                normal_z[axial] = np.where(j1[k][axial] < j0[k][axial], 1.0, -1.0)
                tally["neutral_wall_hits"] += float(w[wl].sum())
                tally["neutral_anode_hits"] += float(w[wl][is_anode].sum())
                tally["neutral_pz_wall"] += float(np.sum(w[wl] * mass * vz[wl]))
                u = rng.random((5, wl.size))
                diffuse = u[0] < self.config.accommodation_coefficient
                # diffuse: cosine law about the inward normal at T_w; specular: flip the normal component
                vth = self.wall_thermal_speed
                vn = flux_maxwellian(u[1], vth)
                g1, g2 = gaussian_pair(u[2], u[3])
                new_vr = np.where(normal_r != 0.0, normal_r * vn, vth * g1)
                new_vt = vth * g2
                new_vz = np.where(normal_z != 0.0, normal_z * vn, vth * g1)
                spec_vr = np.where(normal_r != 0.0, -vr[wl], vr[wl])
                spec_vz = np.where(normal_z != 0.0, -vz[wl], vz[wl])
                vr[wl] = np.where(diffuse, new_vr, spec_vr)
                vt[wl] = np.where(diffuse, new_vt, vt[wl])
                vz[wl] = np.where(diffuse, new_vz, spec_vz)
                tally["neutral_pz_wall"] -= float(np.sum(w[wl] * mass * vz[wl]))
                if self.config.metastables is not None:
                    meta_hit = st[wl] == STATE_METASTABLE
                    deexc = meta_hit & (u[4] < self.config.metastables.wall_deexcitation_probability)
                    tally["meta_wall_deexcited"] += float(w[wl][deexc].sum())
                    st[wl[deexc]] = STATE_GROUND
                # a reflected particle that did not move keeps its remaining time; guard against a zero-velocity stall
                remaining[wl] -= 1e-3 * dt[k]
            active = np.flatnonzero(alive & (remaining > 1e-15 * self.substep_dt_s))
        else:
            raise PIC2DValidationError("neutral march did not converge (a neutral bounced more than the iteration cap)")
        keep = alive
        return NeutralParticles(r[keep], z[keep], vr[keep], vt[keep], vz[keep], w[keep], st[keep])

    def _deposit(self, state: SpatialNeutralState) -> int:
        """Nearest-cell deposit of the ground / metastable atoms and the ground velocity moments; publishes the fields."""

        particles = state.particles
        cells = self._cells(particles)
        ground = particles.state == STATE_GROUND
        w = particles.weight
        sum_g = np.bincount(cells[ground], weights=w[ground], minlength=self.n_cells)
        sum_m = np.bincount(cells[~ground], weights=w[~ground], minlength=self.n_cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            inv_volume = np.where(self.cell_volume > 0.0, 1.0 / np.where(self.cell_volume > 0.0, self.cell_volume, 1.0), 0.0)
            density = sum_g * inv_volume
            meta_density = sum_m * inv_volume
            moments = [np.bincount(cells[ground], weights=(w[ground] * v[ground]), minlength=self.n_cells)
                       for v in (particles.vr_m_per_s, particles.vt_m_per_s, particles.vz_m_per_s)]
            v2 = np.bincount(cells[ground], weights=w[ground] * (particles.vr_m_per_s[ground] ** 2 + particles.vt_m_per_s[ground] ** 2
                                                                  + particles.vz_m_per_s[ground] ** 2), minlength=self.n_cells)
            safe = np.where(sum_g > 0.0, sum_g, 1.0)
            drift = [np.where(sum_g > 0.0, m / safe, 0.0) for m in moments]
            thermal = np.where(sum_g > 0.0, np.sqrt(np.maximum(v2 / safe - (drift[0] ** 2 + drift[1] ** 2 + drift[2] ** 2), 0.0) / 3.0), 0.0)
        violations = int(np.count_nonzero(density > self.ceiling * (1.0 + 1e-12)))
        state.density_per_m3 = np.minimum(density, self.ceiling)
        state.meta_density_per_m3 = meta_density
        state.drift_r, state.drift_t, state.drift_z = drift
        state.thermal_speed = thermal
        return violations

    # ------------------------------------------------------------------ records
    def to_dict(self) -> dict[str, Any]:
        record = {
            **self.config.to_dict(),
            "ceiling_density_per_m3": self.ceiling,
            "neutral_temperature_k": self.temperature_k,
            "wall_temperature_k": self.wall_temperature_k,
            "metastable_macro_weight": self.metastable_weight,
            "substep_dt_neutral_s": self.substep_dt_s,
            "substep_dt_real_s": self.config.substep_steps * self.dt_s,
            "exit_density_zero_ionization_per_m3": self.exit_density,
            "knudsen_anode_over_exit": float(
                knudsen_profile_per_m3(self.geometry, np.array([self.geometry.z_min_m]), self.config.feed_atoms_per_s, self.temperature_k,
                                       mass_kg=self.mass_kg)[0] / self.exit_density),
            "transport": "free-molecular straight flight in path pieces <= 0.5 min(dr, dz); diffuse (cosine law at T_w) / specular wall "
                         "reflection with the accommodation coefficient; anode face reflects; exit aperture / far field removes",
            "depletion": "per-cell weight scaling by the plasma's booked atom sinks (x time_acceleration) with a carried debt; "
                         "recycling / superelastic return / metastable production spawned at the macro-weight granularity with per-cell carries",
            "density_read_by_mcc": "nearest-cell deposit published at every sub-step into a device-resident per-cell array "
                                   "(ground and metastable), clamped at the null-collision ceiling (violations fail closed at the record)",
        }
        return record


__all__ = [
    "METASTABLE_MODEL_V1",
    "NEUTRAL_MODEL_INVENTORY_0D",
    "NEUTRAL_MODEL_SPATIAL_V1",
    "NEUTRAL_RNG_STREAM",
    "NEUTRAL_SPATIAL_LEDGER_KEYS",
    "SINK_FIXED_POINT",
    "STATE_GROUND",
    "STATE_METASTABLE",
    "XE_6S_METASTABLE_EV",
    "CellSinks",
    "MetastableConfig",
    "MetastableProcessTable",
    "NeutralParticles",
    "SpatialNeutralConfig",
    "SpatialNeutralState",
    "SpatialNeutrals",
    "SubstepTally",
    "beb_cross_section_m2",
    "clausing_factor",
    "flux_maxwellian",
    "gaussian_pair",
    "knudsen_profile_per_m3",
    "quantised_fraction",
]
