"""Null-collision Monte Carlo Xe+ - Xe collisions (model v2.3.0 / R3b; CPU reference).

Processes (``spec/pic2d/xenon-ion-neutral-cross-sections-v1.json``, hash-bound):

* ``cex`` - resonant charge exchange (Miller et al. 2002 fit).  The ion continues with the velocity of
  the sampled thermal atom; the atom continues with the ion's velocity as a FAST NEUTRAL, which is not
  a particle of this code.  Its fate is decided at the event from its straight-line flight through the
  cell mask (the same free-molecular assumption the 0-D inventory makes for thermal atoms):
    - slower than ``fast_neutral_speed_threshold_factor x sqrt(k T_g / M)``: a thermal atom -> stays in
      the inventory (``fast_neutral_thermal``);
    - reaches the channel exit plane inside the exit aperture before touching a wall: leaves the
      channel as thrust-carrying neutral flux (``fast_neutral_exit_channel``: the inventory loses the
      atom - the ``fast_neutral_exit`` sink of ``NeutralInventory.advance`` - and its axial momentum and
      kinetic energy are tallied for the thrust ledger);
    - anything else (dielectric wall, cone, anode, or an unresolved march): thermalises on the wall and
      returns to the inventory as a thermal atom (``fast_neutral_wall``: no inventory change; its
      momentum is deposited on the thruster);
    - born downstream of the exit plane (plume domains): leaves the box (``fast_neutral_exit_plume``;
      the plume gas is the effusion cone, not the inventory).
* ``mex`` - momentum transfer, isotropic in the centre of mass (Phelps isotropic component): the
  relative velocity is redirected isotropically, both partners keep their centre-of-mass speeds; the
  atom's recoil is handed to the (0-D) gas.

Null-collision method with a MOVING target: for every ion a candidate is drawn with
``P = 1 - exp(-nu_max dt_i)``; for candidates a Maxwellian atom velocity at ``T_g`` is sampled, the
relative energy ``E = 1/2 M |v_i - v_n|^2`` (the tables' laboratory-frame convention) selects
``sigma_k(E)``, and ``nu_k = n_g sigma_k |v_i - v_n|`` chooses the process against the selector
``u nu_max``.  ``nu_max = n_g0 max_E sum_k sigma_k(E) v(E)`` over the uniform table (the ceiling grows
with energy because sigma_CEX falls only logarithmically); a candidate whose ``sum_k nu_k`` exceeds the
ceiling is counted (``ceiling_violations``) and the simulation fails closed at the next record.

Tallies carry the macro weight ``W`` (the operator knows the ion species): ``energy_loss_j`` is the
kinetic energy the ion population hands to the neutral population, ``pz_ions_kg_m_s`` the ions' axial
momentum change; the fast-neutral terms are what the ledger needs to close momentum and energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import floor, isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .mcc import (
    SPEC_DIR,
    MCCConfig,
    UniformSigmaTable,
    _expm1,
    canonical_payload_sha256,
    isotropic_velocity,
    maxwellian_velocity,
)
from .mesh import MeshMasks
from .models import BOLTZMANN_J_PER_K, EV_J, PIC2DValidationError, ParticleArrays, Species2D

ION_PROCESS_ORDER = ("cex", "mex")
ION_PROCESS_KINDS = {"cex": "charge_exchange", "mex": "momentum_transfer"}
ION_SCHEMA = "cft.pic2d.xenon-ion-neutral-cross-sections.v1"
# cumulative-ledger keys written by both backends (all "extra" keys: absent unless the ion MCC is configured)
ION_MCC_COUNT_KEYS = (
    "ion_mcc_candidates", "ion_mcc_null", "cex", "mex", "cex_plume", "ion_mcc_ceiling_violations",
    "fast_neutral_exit_channel", "fast_neutral_exit_plume", "fast_neutral_wall", "fast_neutral_thermal", "fast_neutral_unresolved",
)
ION_MCC_LEDGER_KEYS = ("ion_neutral_loss_j", "pz_ion_collisions", "pz_fast_neutral_exit", "pz_fast_neutral_wall", "ke_fast_neutral_exit_j")
ION_MCC_KEYS = ION_MCC_COUNT_KEYS + ION_MCC_LEDGER_KEYS
FATE_EXIT = 0
FATE_WALL = 1
FATE_UNRESOLVED = 2


@dataclass(frozen=True, slots=True)
class IonCrossSectionProcess:
    identifier: str
    kind: str
    energy_ev: np.ndarray
    cross_section_m2: np.ndarray
    source: str
    threshold_ev: float = 0.0

    def __post_init__(self) -> None:
        energy = np.asarray(self.energy_ev, dtype=np.float64)
        sigma = np.asarray(self.cross_section_m2, dtype=np.float64)
        if energy.ndim != 1 or energy.shape != sigma.shape or energy.size < 2:
            raise PIC2DValidationError(f"{self.identifier}: tables must be equal-length 1-D arrays")
        if not np.isfinite(energy).all() or not np.isfinite(sigma).all():
            raise PIC2DValidationError(f"{self.identifier}: tables must be finite")
        if np.any(np.diff(energy) <= 0.0) or energy[0] < 0.0:
            raise PIC2DValidationError(f"{self.identifier}: energies must be strictly increasing and >= 0")
        if np.any(sigma < 0.0):
            raise PIC2DValidationError(f"{self.identifier}: cross sections must be non-negative")
        if ION_PROCESS_KINDS.get(self.identifier) != self.kind:
            raise PIC2DValidationError(f"{self.identifier}: unsupported ion process kind {self.kind!r}")
        if self.threshold_ev != 0.0:
            raise PIC2DValidationError(f"{self.identifier}: ion-neutral processes are resonant / elastic (threshold 0)")
        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(self, "cross_section_m2", sigma)

    def at(self, energy_ev: np.ndarray) -> np.ndarray:
        e = np.asarray(energy_ev, dtype=np.float64)
        return np.interp(e, self.energy_ev, self.cross_section_m2)


@dataclass(frozen=True, slots=True)
class IonNeutralCrossSections:
    """Validated Xe+ / Xe set (``cex``, ``mex``) with a bound payload hash."""

    processes: tuple[IonCrossSectionProcess, ...]
    provenance: Mapping[str, Any]
    payload_sha256: str
    file_sha256: str | None

    def __post_init__(self) -> None:
        identifiers = tuple(process.identifier for process in self.processes)
        if identifiers != ION_PROCESS_ORDER:
            raise PIC2DValidationError(f"ion processes must be exactly {ION_PROCESS_ORDER} in order, got {identifiers}")

    @classmethod
    def from_file(cls, path: Path) -> "IonNeutralCrossSections":
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        if document.get("schema") != ION_SCHEMA:
            raise PIC2DValidationError("unsupported ion-neutral cross-section schema")
        integrity = document.get("integrity", {})
        payload = {key: value for key, value in document.items() if key != "integrity"}
        digest = canonical_payload_sha256(payload)
        if integrity.get("algorithm") != "sha256" or integrity.get("payload_sha256") != digest:
            raise PIC2DValidationError("ion-neutral cross-section payload SHA-256 does not match its integrity record")
        if document.get("units") != {"energy": "eV", "cross_section": "m2"} or document.get("projectile") != "Xe+":
            raise PIC2DValidationError("ion-neutral cross sections must be Xe+ on Xe in eV and m2")
        processes = tuple(
            IonCrossSectionProcess(str(item["id"]), str(item["kind"]), np.asarray(item["energy_ev"], dtype=np.float64),
                                   np.asarray(item["cross_section_m2"], dtype=np.float64), str(item.get("source", "")),
                                   float(item.get("threshold_ev", 0.0)))
            for item in document["processes"]
        )
        return cls(processes, dict(document["provenance"]), digest, sha256(raw).hexdigest())

    @classmethod
    def synthetic_for_tests(cls, *, cex_m2: float = 5.0e-19, mex_m2: float = 1.0e-19) -> "IonNeutralCrossSections":
        """Energy-independent (hard-sphere) tables for numerics-only tests (NOT xenon data)."""

        energy = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 2000.0])
        processes = (
            IonCrossSectionProcess("cex", "charge_exchange", energy, np.full(energy.size, cex_m2), "synthetic-verification:pic2d"),
            IonCrossSectionProcess("mex", "momentum_transfer", energy, np.full(energy.size, mex_m2), "synthetic-verification:pic2d"),
        )
        provenance = {"status": "synthetic-verification", "notes": "numerics-only fixture; not xenon data"}
        return cls(processes, provenance, canonical_payload_sha256({"synthetic-ion": [cex_m2, mex_m2]}), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "provenance_status": self.provenance.get("status"),
            "processes": [{"id": p.identifier, "kind": p.kind, "points": int(p.energy_ev.size), "source": p.source} for p in self.processes],
        }


@dataclass(frozen=True, slots=True)
class IonNeutralMCCConfig:
    """Declared ion-neutral block of a collision set (part of ``config_sha256``)."""

    file: str
    payload_sha256: str
    processes: tuple[tuple[str, str], ...]
    energy_step_ev: float = 0.05
    energy_max_ev: float = 2000.0
    fast_neutral_speed_threshold_factor: float = 4.0

    def __post_init__(self) -> None:
        if not isinstance(self.payload_sha256, str) or len(self.payload_sha256) != 64:
            raise PIC2DValidationError("ion-neutral payload_sha256 must be a 64-hex sha256")
        if tuple(p[0] for p in self.processes) != ION_PROCESS_ORDER:
            raise PIC2DValidationError(f"ion-neutral processes must be {ION_PROCESS_ORDER}")
        if not 0.0 < self.energy_step_ev <= 1.0 or not 100.0 <= self.energy_max_ev <= 1.0e5:
            raise PIC2DValidationError("ion cross-section grid parameters are out of range")
        if not isfinite(self.fast_neutral_speed_threshold_factor) or self.fast_neutral_speed_threshold_factor < 0.0:
            raise PIC2DValidationError("fast_neutral_speed_threshold_factor must be finite and >= 0")
        object.__setattr__(self, "processes", tuple((str(i), str(k)) for i, k in self.processes))

    def load(self) -> IonNeutralCrossSections:
        path = Path(self.file)
        cross_sections = IonNeutralCrossSections.from_file(path if path.is_absolute() else SPEC_DIR / path)
        if cross_sections.payload_sha256 != self.payload_sha256:
            raise PIC2DValidationError(f"{self.file}: payload sha256 {cross_sections.payload_sha256[:12]} differs from the declared {self.payload_sha256[:12]}")
        return cross_sections

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "payload_sha256": self.payload_sha256,
            "processes": [{"id": i, "kind": k} for i, k in self.processes],
            "energy_step_ev": self.energy_step_ev,
            "energy_max_ev": self.energy_max_ev,
            "fast_neutral_speed_threshold_factor": self.fast_neutral_speed_threshold_factor,
            "kinematics": {
                "cex": "velocity swap with a Maxwellian atom at T_g; the atom leaves as a fast neutral (fate: thermal / exit / wall by straight-line flight through the cell mask)",
                "mex": "isotropic redirection of the relative velocity in the centre of mass; the atom's recoil goes to the 0-D gas",
                "energy_argument": "E = 1/2 M_Xe |v_ion - v_atom|^2 (ion energy for a stationary atom)",
            },
        }


def relative_speed_from_energy(energy_ev: np.ndarray, mass_kg: float) -> np.ndarray:
    return np.sqrt(2.0 * np.asarray(energy_ev, dtype=np.float64) * EV_J / mass_kg)


def ion_maximum_collision_frequency(table: UniformSigmaTable, neutral_density_per_m3: float, mass_kg: float) -> float:
    """``nu_max = n_g max_E [sum_k sigma_k(E) g(E)]`` with ``g = sqrt(2 E / M)`` over the uniform grid."""

    energy = np.arange(table.point_count, dtype=np.float64) * table.energy_step_ev
    total = table.table_m2.sum(axis=0) * relative_speed_from_energy(energy, mass_kg)
    return float(neutral_density_per_m3 * np.max(total))


def fast_neutral_fate(masks: MeshMasks, r_m: float, z_m: float, vr: float, vt: float, vz: float, *, max_steps: int | None = None) -> int:
    """Straight-line flight of a fast neutral born at ``(r, z)`` through the cell mask (channel zone).

    Marches in path-length steps of ``0.5 min(dr, dz)`` (``r(t) = sqrt((r + v_r t)^2 + (v_theta t)^2)``, ``z(t) = z + v_z t``)
    until it crosses the channel exit plane (``FATE_EXIT`` when inside the exit aperture, else ``FATE_WALL``), the anode plane,
    the outer box or a non-plasma cell (``FATE_WALL``).  The Warp kernel repeats this arithmetic step for step.
    """

    grid = masks.grid
    geometry = grid.geometry
    dr, dz = grid.dr_m, grid.dz_m
    nr, nz = grid.cell_shape
    z_min, z_exit, r_exit = geometry.z_min_m, geometry.z_max_m, geometry.exit_radius_m
    if z_m >= z_exit:
        return FATE_EXIT
    speed = sqrt(vr * vr + vt * vt + vz * vz)
    if speed <= 0.0:
        return FATE_WALL
    step_dt = 0.5 * min(dr, dz) / speed
    limit = 4 * (nr + nz) + 8 if max_steps is None else int(max_steps)
    plasma = masks.plasma_cell
    for k in range(1, limit + 1):
        t = k * step_dt
        x = r_m + vr * t
        y = vt * t
        rr = sqrt(x * x + y * y)
        zz = z_m + vz * t
        if zz >= z_exit:
            return FATE_EXIT if rr < r_exit else FATE_WALL
        if zz < z_min:
            return FATE_WALL
        i = int(floor(rr / dr))
        if i >= nr:
            return FATE_WALL
        j = min(max(int(floor((zz - z_min) / dz)), 0), nz - 1)
        if not plasma[i, j]:
            return FATE_WALL
    return FATE_UNRESOLVED


@dataclass(frozen=True, slots=True)
class IonMCCTally:
    """Per-step ion-neutral tallies; counts are macro events, energies / momenta carry ``W``."""

    candidates: int = 0
    cex: int = 0
    mex: int = 0
    null: int = 0
    cex_plume: int = 0
    ceiling_violations: int = 0
    fast_neutral_exit_channel: int = 0
    fast_neutral_exit_plume: int = 0
    fast_neutral_wall: int = 0
    fast_neutral_thermal: int = 0
    fast_neutral_unresolved: int = 0
    energy_loss_j: float = 0.0
    pz_ions_kg_m_s: float = 0.0
    pz_fast_neutral_exit_kg_m_s: float = 0.0
    pz_fast_neutral_wall_kg_m_s: float = 0.0
    ke_fast_neutral_exit_j: float = 0.0

    def to_cumulative(self) -> dict[str, float]:
        return {
            "ion_mcc_candidates": float(self.candidates), "ion_mcc_null": float(self.null), "cex": float(self.cex), "mex": float(self.mex),
            "cex_plume": float(self.cex_plume), "ion_mcc_ceiling_violations": float(self.ceiling_violations),
            "fast_neutral_exit_channel": float(self.fast_neutral_exit_channel), "fast_neutral_exit_plume": float(self.fast_neutral_exit_plume),
            "fast_neutral_wall": float(self.fast_neutral_wall), "fast_neutral_thermal": float(self.fast_neutral_thermal),
            "fast_neutral_unresolved": float(self.fast_neutral_unresolved),
            "ion_neutral_loss_j": self.energy_loss_j, "pz_ion_collisions": self.pz_ions_kg_m_s,
            "pz_fast_neutral_exit": self.pz_fast_neutral_exit_kg_m_s, "pz_fast_neutral_wall": self.pz_fast_neutral_wall_kg_m_s,
            "ke_fast_neutral_exit_j": self.ke_fast_neutral_exit_j,
        }


@dataclass(frozen=True, slots=True)
class IonMCCResult:
    ions: ParticleArrays
    tally: IonMCCTally


class IonNullCollisionMCC:
    """CPU reference null-collision operator for Xe+ on the thermal neutral background."""

    def __init__(self, cross_sections: IonNeutralCrossSections, mcc_config: MCCConfig, ion_config: IonNeutralMCCConfig,
                 ion_species: Species2D, masks: MeshMasks) -> None:
        if cross_sections.payload_sha256 != ion_config.payload_sha256:
            raise PIC2DValidationError("ion-neutral cross sections differ from the declared payload hash")
        self.cross_sections = cross_sections
        self.mcc_config = mcc_config
        self.config = ion_config
        self.ion_species = ion_species
        self.masks = masks
        self.table = UniformSigmaTable.build(cross_sections, energy_step_ev=ion_config.energy_step_ev, energy_max_ev=ion_config.energy_max_ev)  # type: ignore[arg-type]
        self.nu_max = ion_maximum_collision_frequency(self.table, mcc_config.neutral_density_per_m3, ion_species.mass_kg)
        self.neutral_scale = 1.0
        self.thermal_speed = sqrt(BOLTZMANN_J_PER_K * mcc_config.neutral_temperature_k / ion_species.mass_kg)
        self.fast_speed_threshold = ion_config.fast_neutral_speed_threshold_factor * self.thermal_speed

    def set_neutral_scale(self, scale: float) -> None:
        if not isfinite(scale) or scale < 0.0 or scale > 1.0 + 1e-9:
            raise PIC2DValidationError(f"neutral scale {scale!r} must lie in [0, 1] (null-collision ceiling)")
        self.neutral_scale = float(scale)

    @property
    def neutral_density_per_m3(self) -> float:
        return self.mcc_config.neutral_density_per_m3 * self.neutral_scale

    def collision_probability(self, dt_s: float) -> float:
        return -_expm1(-self.nu_max * dt_s)

    def apply(self, ions: ParticleArrays, dt_s: float, rng: np.random.Generator, *, density_shape: np.ndarray | None = None) -> IonMCCResult:
        count = ions.count
        probability = self.collision_probability(dt_s)
        if count == 0 or probability == 0.0:
            return IonMCCResult(ions, IonMCCTally())
        mass = self.ion_species.mass_kg
        weight = self.ion_species.macro_weight
        u = rng.random((8, count))
        candidate = u[0] < probability
        idx = np.flatnonzero(candidate)
        vx = ions.vr_m_per_s.copy()
        vy = ions.vt_m_per_s.copy()
        vz = ions.vz_m_per_s.copy()
        tally = IonMCCTally(candidates=int(idx.size))
        if idx.size == 0:
            return IonMCCResult(ParticleArrays(ions.r_m.copy(), ions.z_m.copy(), vx, vy, vz), tally)
        # thermal atom velocity for every candidate (Box-Muller, four uniforms)
        nvx, nvy, nvz = maxwellian_velocity(mass, self.mcc_config.neutral_temperature_k, u[2:6][:, idx])
        gx, gy, gz = vx[idx] - nvx, vy[idx] - nvy, vz[idx] - nvz
        g2 = gx * gx + gy * gy + gz * gz
        g = np.sqrt(g2)
        energy = 0.5 * mass * g2 / EV_J
        sigma = self.table.lookup(energy)             # (2, N_c)
        density = self.neutral_density_per_m3
        if density_shape is not None:
            shape = np.asarray(density_shape, dtype=np.float64)
            if shape.shape != (count,) or not np.isfinite(shape).all() or np.any(shape < 0.0) or np.any(shape > 1.0 + 1e-12):
                raise PIC2DValidationError("neutral density shape must be per particle in [0, 1]")
            density = density * shape[idx]
        nu_cex = density * sigma[0] * g
        nu_mex = density * sigma[1] * g
        total = nu_cex + nu_mex
        violations = int(np.count_nonzero(total > self.nu_max * (1.0 + 1e-12)))
        selector = u[1][idx] * self.nu_max
        is_cex = selector < nu_cex
        is_mex = ~is_cex & (selector < total)
        null = int(np.count_nonzero(~(is_cex | is_mex)))
        old_vx, old_vy, old_vz = vx[idx].copy(), vy[idx].copy(), vz[idx].copy()
        new_vx, new_vy, new_vz = old_vx.copy(), old_vy.copy(), old_vz.copy()
        # CEX: velocity swap
        new_vx[is_cex], new_vy[is_cex], new_vz[is_cex] = nvx[is_cex], nvy[is_cex], nvz[is_cex]
        # MEX: isotropic in the centre of mass (equal masses: v_cm = (v_i + v_n) / 2, v_i' = v_cm + g' / 2)
        if np.any(is_mex):
            gpx, gpy, gpz = isotropic_velocity(g[is_mex], u[6][idx][is_mex], u[7][idx][is_mex])
            new_vx[is_mex] = 0.5 * (old_vx[is_mex] + nvx[is_mex]) + 0.5 * gpx
            new_vy[is_mex] = 0.5 * (old_vy[is_mex] + nvy[is_mex]) + 0.5 * gpy
            new_vz[is_mex] = 0.5 * (old_vz[is_mex] + nvz[is_mex]) + 0.5 * gpz
        changed = is_cex | is_mex
        ke_before = 0.5 * mass * (old_vx * old_vx + old_vy * old_vy + old_vz * old_vz)
        ke_after = 0.5 * mass * (new_vx * new_vx + new_vy * new_vy + new_vz * new_vz)
        energy_loss = weight * float(np.sum((ke_before - ke_after)[changed]))
        pz_ions = weight * mass * float(np.sum((new_vz - old_vz)[changed]))
        vx[idx], vy[idx], vz[idx] = new_vx, new_vy, new_vz
        # fast-neutral fate for every CEX event
        z_exit = self.masks.grid.geometry.z_max_m
        exit_channel = exit_plume = wall = thermal = unresolved = cex_plume = 0
        pz_exit = pz_wall = ke_exit = 0.0
        for k in np.flatnonzero(is_cex):
            p = idx[k]
            fvx, fvy, fvz = old_vx[k], old_vy[k], old_vz[k]
            speed = sqrt(fvx * fvx + fvy * fvy + fvz * fvz)
            in_plume = ions.z_m[p] >= z_exit
            if in_plume:
                cex_plume += 1
            if speed < self.fast_speed_threshold:
                thermal += 1
                continue
            if in_plume:
                exit_plume += 1
                pz_exit += mass * fvz
                ke_exit += 0.5 * mass * speed * speed
                continue
            fate = fast_neutral_fate(self.masks, float(ions.r_m[p]), float(ions.z_m[p]), float(fvx), float(fvy), float(fvz))
            if fate == FATE_EXIT:
                exit_channel += 1
                pz_exit += mass * fvz
                ke_exit += 0.5 * mass * speed * speed
            else:
                wall += 1
                pz_wall += mass * fvz
                if fate == FATE_UNRESOLVED:
                    unresolved += 1
        tally = IonMCCTally(
            candidates=int(idx.size), cex=int(np.count_nonzero(is_cex)), mex=int(np.count_nonzero(is_mex)), null=null, cex_plume=cex_plume,
            ceiling_violations=violations, fast_neutral_exit_channel=exit_channel, fast_neutral_exit_plume=exit_plume,
            fast_neutral_wall=wall, fast_neutral_thermal=thermal, fast_neutral_unresolved=unresolved,
            energy_loss_j=energy_loss, pz_ions_kg_m_s=pz_ions, pz_fast_neutral_exit_kg_m_s=weight * pz_exit,
            pz_fast_neutral_wall_kg_m_s=weight * pz_wall, ke_fast_neutral_exit_j=weight * ke_exit,
        )
        return IonMCCResult(ParticleArrays(ions.r_m.copy(), ions.z_m.copy(), vx, vy, vz), tally)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "cross_sections": self.cross_sections.to_dict(),
            "uniform_table_sha256": self.table.sha256(),
            "nu_max_per_s": self.nu_max,
            "neutral_temperature_k": self.mcc_config.neutral_temperature_k,
            "fast_neutral_speed_threshold_m_per_s": self.fast_speed_threshold,
            "fast_neutral_fate": (
                "straight-line flight through the cell mask in 0.5 min(dr, dz) steps: exit aperture -> leaves (inventory sink, thrust "
                "ledger); wall / cone / anode / unresolved -> thermalised on the wall (inventory unchanged, momentum to the thruster); "
                "below the speed threshold -> thermal atom (inventory unchanged); plume-born -> leaves the box"
            ),
        }


__all__ = [
    "FATE_EXIT",
    "FATE_UNRESOLVED",
    "FATE_WALL",
    "ION_MCC_COUNT_KEYS",
    "ION_MCC_KEYS",
    "ION_MCC_LEDGER_KEYS",
    "ION_PROCESS_ORDER",
    "ION_SCHEMA",
    "IonCrossSectionProcess",
    "IonMCCResult",
    "IonMCCTally",
    "IonNeutralCrossSections",
    "IonNeutralMCCConfig",
    "IonNullCollisionMCC",
    "fast_neutral_fate",
    "ion_maximum_collision_frequency",
    "relative_speed_from_energy",
]
