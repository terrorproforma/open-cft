"""Null-collision Monte Carlo electron–xenon collisions (CPU reference).

Processes (v1): elastic (isotropic, speed preserved: the ``2 m_e/M`` energy
transfer to the atom is neglected), one lumped excitation (threshold 8.32 eV,
isotropic), single ionisation (threshold 12.13 eV; secondary energy from the
Vahedi–Surendra distribution ``E_s = B tan[xi atan((E - E_iz)/(2B))]`` with
``B = 8.7 eV`` for xenon, isotropic emission of both electrons; the ion is born
at the event position with a Maxwellian neutral velocity at ``T_g``).  Neutrals
are a uniform, static background: ion–neutral collisions and neutral depletion
are not modelled in v1.

Cross sections are resampled once onto a uniform energy grid so that CPU and
GPU evaluate bit-identical ``sigma(E)``; only the random streams differ.

Model v2.3.0 (``xe_collision_set_v2``, 2026-09-05): the process list is ``elastic``, then ONE OR
MORE excitation levels (each with its own threshold = the energy removed per event), then
``ionization``.  The legacy three-process set (``PROCESS_ORDER``) is the one-level special case
and its arithmetic and random-number consumption are unchanged (bitwise v2.0.6 replay); the
multi-level set is loaded through ``cft_revival.pic2d.cross_sections_xe`` and selected by
``MCCConfig.collision_set`` (which is what enters ``config_sha256``).  Ion-neutral collisions
live in ``cft_revival.pic2d.ion_mcc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import exp, isfinite, pi, sqrt
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .models import (
    BOLTZMANN_J_PER_K,
    ELECTRON_MASS_KG,
    EV_J,
    PIC2DValidationError,
    ParticleArrays,
    Species2D,
)

SPEC_DIR = Path(__file__).resolve().parents[3] / "spec" / "pic2d"
DEFAULT_CROSS_SECTION_PATH = SPEC_DIR / "xenon-cross-sections-v1.json"
PROCESS_ORDER = ("elastic", "excitation", "ionization")
CROSS_SECTION_SCHEMAS = ("cft.pic2d.xenon-cross-sections.v1", "cft.pic2d.xenon-cross-sections.v2")
MAX_EXCITATION_LEVELS = 8      # device tally slots reserved per level (the Warp backend)
VAHEDI_SURENDRA_B_EV = 8.7


def validate_process_order(processes: tuple["CrossSectionProcess", ...]) -> None:
    """``elastic``, then >= 1 excitation levels (ascending thresholds), then ``ionization`` (fail closed)."""

    identifiers = tuple(process.identifier for process in processes)
    kinds = tuple(process.kind for process in processes)
    if identifiers == PROCESS_ORDER and kinds == PROCESS_ORDER:
        return
    if len(processes) < 3 or identifiers[0] != "elastic" or kinds[0] != "elastic" or identifiers[-1] != "ionization" or kinds[-1] != "ionization":
        raise PIC2DValidationError(f"processes must be exactly {PROCESS_ORDER} or elastic, excitation levels..., ionization; got {identifiers}")
    levels = processes[1:-1]
    if any(process.kind != "excitation" for process in levels) or len(levels) > MAX_EXCITATION_LEVELS:
        raise PIC2DValidationError(f"the middle processes must be 1..{MAX_EXCITATION_LEVELS} excitation levels, got {identifiers[1:-1]}")
    thresholds = [process.threshold_ev for process in levels]
    if any(b <= a for a, b in zip(thresholds, thresholds[1:])) or len(set(identifiers)) != len(identifiers):
        raise PIC2DValidationError("excitation levels must have strictly increasing thresholds and unique identifiers")


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CrossSectionProcess:
    identifier: str
    kind: str
    threshold_ev: float
    energy_ev: np.ndarray
    cross_section_m2: np.ndarray
    source: str

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
        if not isfinite(self.threshold_ev) or self.threshold_ev < 0.0:
            raise PIC2DValidationError(f"{self.identifier}: threshold must be finite and >= 0")
        if self.kind not in {"elastic", "excitation", "ionization"}:
            raise PIC2DValidationError(f"{self.identifier}: unsupported process kind {self.kind!r}")
        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(self, "cross_section_m2", sigma)

    def at(self, energy_ev: np.ndarray) -> np.ndarray:
        e = np.asarray(energy_ev, dtype=np.float64)
        values = np.interp(e, self.energy_ev, self.cross_section_m2)
        return np.where(e < self.threshold_ev, 0.0, values)


@dataclass(frozen=True, slots=True)
class XenonCrossSections:
    """Validated e–Xe cross-section set with a bound source hash."""

    processes: tuple[CrossSectionProcess, ...]
    provenance: Mapping[str, Any]
    payload_sha256: str
    file_sha256: str | None

    def __post_init__(self) -> None:
        validate_process_order(self.processes)

    @property
    def excitation_levels(self) -> tuple[CrossSectionProcess, ...]:
        """The excitation processes in table order (one for the legacy lumped set)."""

        return self.processes[1:-1]

    @property
    def is_legacy_set(self) -> bool:
        return tuple(process.identifier for process in self.processes) == PROCESS_ORDER

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CROSS_SECTION_PATH) -> "XenonCrossSections":
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        if document.get("schema") not in CROSS_SECTION_SCHEMAS:
            raise PIC2DValidationError("unsupported cross-section schema")
        integrity = document.get("integrity", {})
        payload = {key: value for key, value in document.items() if key != "integrity"}
        digest = canonical_payload_sha256(payload)
        if integrity.get("algorithm") != "sha256" or integrity.get("payload_sha256") != digest:
            raise PIC2DValidationError("cross-section payload SHA-256 does not match its integrity record")
        if document.get("units") != {"energy": "eV", "cross_section": "m2"}:
            raise PIC2DValidationError("cross-section units must be eV and m2")
        processes = tuple(
            CrossSectionProcess(
                str(item["id"]), str(item["kind"]), float(item["threshold_ev"]),
                np.asarray(item["energy_ev"], dtype=np.float64),
                np.asarray(item["cross_section_m2"], dtype=np.float64),
                str(item.get("source", "")),
            )
            for item in document["processes"]
        )
        return cls(processes, dict(document["provenance"]), digest, sha256(raw).hexdigest())

    @classmethod
    def synthetic_for_tests(cls) -> "XenonCrossSections":
        """Smooth synthetic tables (NOT xenon data) for numerics-only tests."""

        energy = np.concatenate(([0.0], np.geomspace(0.01, 2000.0, 120)))
        elastic = 2.0e-19 * np.exp(-((np.log10(np.maximum(energy, 1e-3)) - 1.0) ** 2) / 1.5) + 2.0e-20
        excitation = np.where(energy > 8.32, 1.5e-20 * (1.0 - np.exp(-(energy - 8.32) / 15.0)) * 30.0 / (30.0 + energy), 0.0)
        ionization = np.where(energy > 12.13, 4.5e-20 * (1.0 - np.exp(-(energy - 12.13) / 40.0)) * 100.0 / (100.0 + energy), 0.0)
        processes = (
            CrossSectionProcess("elastic", "elastic", 0.0, energy, elastic, "synthetic-verification:pic2d"),
            CrossSectionProcess("excitation", "excitation", 8.32, energy, excitation, "synthetic-verification:pic2d"),
            CrossSectionProcess("ionization", "ionization", 12.13, energy, ionization, "synthetic-verification:pic2d"),
        )
        provenance = {"status": "synthetic-verification", "notes": "numerics-only fixture; not xenon data"}
        return cls(processes, provenance, canonical_payload_sha256({"synthetic": True}), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_sha256": self.payload_sha256,
            "file_sha256": self.file_sha256,
            "provenance_status": self.provenance.get("status"),
            "processes": [
                {"id": p.identifier, "kind": p.kind, "threshold_ev": p.threshold_ev, "points": int(p.energy_ev.size), "source": p.source}
                for p in self.processes
            ],
        }


@dataclass(frozen=True, slots=True)
class UniformSigmaTable:
    """Cross sections resampled on a uniform energy grid shared by CPU and GPU.

    Rows follow the process order: ``0`` elastic, ``1 .. n_exc`` the excitation levels, ``n_exc + 1``
    ionisation (``(3, n_points)`` for the legacy set).
    """

    energy_step_ev: float
    energy_max_ev: float
    table_m2: np.ndarray  # shape (n_processes, n_points)
    thresholds_ev: tuple[float, ...]

    @classmethod
    def build(cls, cross_sections: XenonCrossSections, *, energy_step_ev: float = 0.05, energy_max_ev: float = 2000.0) -> "UniformSigmaTable":
        count = int(round(energy_max_ev / energy_step_ev)) + 1
        grid = np.arange(count, dtype=np.float64) * energy_step_ev
        table = np.stack([process.at(grid) for process in cross_sections.processes])
        thresholds = tuple(float(process.threshold_ev) for process in cross_sections.processes)
        return cls(float(energy_step_ev), float(grid[-1]), table, thresholds)

    @property
    def point_count(self) -> int:
        return int(self.table_m2.shape[1])

    @property
    def process_count(self) -> int:
        return int(self.table_m2.shape[0])

    @property
    def excitation_count(self) -> int:
        return self.process_count - 2

    @property
    def ionization_row(self) -> int:
        return self.process_count - 1

    @property
    def excitation_thresholds_ev(self) -> tuple[float, ...]:
        return self.thresholds_ev[1:-1]

    def lookup(self, energy_ev: np.ndarray) -> np.ndarray:
        """Piecewise-linear lookup; returns shape (n_processes, N).  Clamps above the grid."""

        e = np.clip(np.asarray(energy_ev, dtype=np.float64), 0.0, self.energy_max_ev)
        position = e / self.energy_step_ev
        index = np.minimum(np.floor(position).astype(np.int64), self.point_count - 2)
        fraction = position - index
        lower = self.table_m2[:, index]
        upper = self.table_m2[:, index + 1]
        return lower + fraction * (upper - lower)

    def sha256(self) -> str:
        digest = sha256()
        digest.update(np.asarray([self.energy_step_ev, self.energy_max_ev], dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(self.table_m2).tobytes())
        return digest.hexdigest()


def electron_speed_from_energy(energy_ev: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 * np.asarray(energy_ev, dtype=np.float64) * EV_J / ELECTRON_MASS_KG)


def electron_energy_ev(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    return 0.5 * ELECTRON_MASS_KG * (vx * vx + vy * vy + vz * vz) / EV_J


def maximum_collision_frequency(table: UniformSigmaTable, neutral_density_per_m3: float) -> float:
    """``nu_max = n_g max_E [sum_k sigma_k(E) v(E)]`` over the uniform grid."""

    energy = np.arange(table.point_count, dtype=np.float64) * table.energy_step_ev
    speed = electron_speed_from_energy(energy)
    total = table.table_m2.sum(axis=0) * speed
    return float(neutral_density_per_m3 * np.max(total))


@dataclass(frozen=True, slots=True)
class MCCConfig:
    neutral_density_per_m3: float
    neutral_temperature_k: float = 300.0
    energy_step_ev: float = 0.05
    energy_max_ev: float = 2000.0
    # v2.3.0: the declared collision set (``cross_sections_xe.CollisionSetConfig``): the electron set's payload hash and
    # process list plus the optional ion-neutral (CEX / MEX) block.  None = the legacy v1 lumped set with collisionless
    # ions; its identity (``to_dict``) is unchanged.  Duck-typed (``to_dict``, ``electron_payload_sha256``, ``ion_neutral``)
    # to keep this module free of the collision-set module.
    collision_set: Any | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.neutral_density_per_m3) or self.neutral_density_per_m3 < 0.0:
            raise PIC2DValidationError("neutral density must be finite and non-negative")
        if not isfinite(self.neutral_temperature_k) or self.neutral_temperature_k <= 0.0:
            raise PIC2DValidationError("neutral temperature must be positive")
        if not 0.0 < self.energy_step_ev <= 1.0 or not 100.0 <= self.energy_max_ev <= 1.0e5:
            raise PIC2DValidationError("cross-section grid parameters are out of range")
        if self.collision_set is not None and not (hasattr(self.collision_set, "to_dict") and hasattr(self.collision_set, "electron_payload_sha256")):
            raise PIC2DValidationError("collision_set must be a CollisionSetConfig (to_dict + electron_payload_sha256)")

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "neutral_density_per_m3": self.neutral_density_per_m3,
            "neutral_temperature_k": self.neutral_temperature_k,
            "energy_step_ev": self.energy_step_ev,
            "energy_max_ev": self.energy_max_ev,
        }
        if self.collision_set is not None:     # v2.3.0: present only when declared -> every earlier identity is unchanged
            record["collision_set"] = self.collision_set.to_dict()
        return record


@dataclass(frozen=True, slots=True)
class MCCTally:
    """Per-step event counts of the null-collision operator (MACRO-particle events).

    ``inelastic_energy_loss_j`` is the threshold energy removed from the colliding macro-electrons,
    ``(n_exc E_exc + n_ion E_ion) e`` - per macro event, i.e. WITHOUT the macro weight ``W`` (the
    operator does not know it).  The simulation's energy ledger multiplies it by ``W`` before it
    enters ``cumulative["inelastic_loss_j"]`` (model v2.0.6, 2026-09-05: up to v2.0.5 the ledger
    took this number as it is, so every recorded residual was biased negative by the inelastic
    power; the unscaled sum is kept as ``inelastic_loss_per_weight_j`` for continuity).
    """

    candidates: int
    elastic: int
    excitation: int
    ionization: int
    null: int
    inelastic_energy_loss_j: float
    # v2.3.0: per-level excitation counts in table order (``sum == excitation``; one entry for the legacy set)
    excitation_levels: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "candidates": self.candidates,
            "elastic": self.elastic,
            "excitation": self.excitation,
            "ionization": self.ionization,
            "null": self.null,
            "inelastic_energy_loss_j": self.inelastic_energy_loss_j,
        }
        if len(self.excitation_levels) > 1:
            record["excitation_levels"] = list(self.excitation_levels)
        return record


@dataclass(frozen=True, slots=True)
class MCCResult:
    electrons: ParticleArrays
    new_electrons: ParticleArrays
    new_ions: ParticleArrays
    ionization_r_m: np.ndarray
    ionization_z_m: np.ndarray
    tally: MCCTally


def isotropic_velocity(speed: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_chi = 1.0 - 2.0 * u1
    sin_chi = np.sqrt(np.maximum(0.0, 1.0 - cos_chi * cos_chi))
    phi = 2.0 * pi * u2
    return speed * sin_chi * np.cos(phi), speed * sin_chi * np.sin(phi), speed * cos_chi


def maxwellian_velocity(mass_kg: float, temperature_k: float, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Box–Muller Maxwellian from four uniforms per particle (shape (4, N))."""

    thermal = sqrt(BOLTZMANN_J_PER_K * temperature_k / mass_kg)
    radius1 = np.sqrt(-2.0 * np.log(np.maximum(u[0], 1e-300)))
    radius2 = np.sqrt(-2.0 * np.log(np.maximum(u[2], 1e-300)))
    vx = thermal * radius1 * np.cos(2.0 * pi * u[1])
    vy = thermal * radius1 * np.sin(2.0 * pi * u[1])
    vz = thermal * radius2 * np.cos(2.0 * pi * u[3])
    return vx, vy, vz


class NullCollisionMCC:
    """CPU reference null-collision operator for electrons on a static neutral background."""

    def __init__(self, cross_sections: XenonCrossSections, config: MCCConfig, ion_species: Species2D) -> None:
        self.cross_sections = cross_sections
        self.config = config
        self.ion_species = ion_species
        self.table = UniformSigmaTable.build(
            cross_sections, energy_step_ev=config.energy_step_ev, energy_max_ev=config.energy_max_ev
        )
        self.nu_max = maximum_collision_frequency(self.table, config.neutral_density_per_m3)
        # v1.3: the instantaneous neutral density is ``neutral_scale * config.neutral_density_per_m3``;
        # the null-collision ceiling ``nu_max`` stays at the configured density, so the scale
        # must not exceed 1 (fail closed).
        self.neutral_scale = 1.0

    def set_neutral_scale(self, scale: float) -> None:
        if not isfinite(scale) or scale < 0.0 or scale > 1.0 + 1e-9:
            raise PIC2DValidationError(f"neutral scale {scale!r} must lie in [0, 1] (null-collision ceiling)")
        self.neutral_scale = float(scale)

    @property
    def neutral_density_per_m3(self) -> float:
        return self.config.neutral_density_per_m3 * self.neutral_scale

    def collision_probability(self, dt_s: float) -> float:
        return -_expm1(-self.nu_max * dt_s)

    def apply(
        self, electrons: ParticleArrays, dt_s: float, rng: np.random.Generator, *, density_shape: np.ndarray | None = None,
    ) -> MCCResult:
        """One null-collision step.

        ``density_shape`` (v2.0, optional, per particle in [0, 1]) multiplies the neutral
        density at each electron's position (the two-zone channel/plume field); the null
        ceiling stays at the configured density, so the shape must not exceed 1.
        """

        count = electrons.count
        probability = self.collision_probability(dt_s)
        if count == 0 or probability == 0.0:
            empty = ParticleArrays.empty()
            return MCCResult(electrons, empty, empty, np.zeros(0), np.zeros(0), MCCTally(0, 0, 0, 0, 0, 0.0))
        u = rng.random((12, count))
        candidate = u[0] < probability
        n_candidates = int(np.count_nonzero(candidate))
        vx = electrons.vr_m_per_s.copy()
        vy = electrons.vt_m_per_s.copy()
        vz = electrons.vz_m_per_s.copy()
        energy = electron_energy_ev(vx, vy, vz)
        speed = np.sqrt(vx * vx + vy * vy + vz * vz)
        sigma = self.table.lookup(energy)  # (3, N)
        density = self.neutral_density_per_m3
        if density_shape is not None:
            shape = np.asarray(density_shape, dtype=np.float64)
            if shape.shape != (count,) or not np.isfinite(shape).all() or np.any(shape < 0.0) or np.any(shape > 1.0 + 1e-12):
                raise PIC2DValidationError("neutral density shape must be per particle in [0, 1]")
            density = density * shape
        nu = density * sigma * speed[None, :]
        cumulative = np.cumsum(nu, axis=0)
        selector = u[1] * self.nu_max
        # process k wins when the selector falls in [cumulative[k-1], cumulative[k]) (the null-collision rule); for the
        # legacy 3-row table this is exactly the v1 elastic / excitation / ionization chain
        assigned = ~candidate
        selected: list[np.ndarray] = []
        for k in range(cumulative.shape[0]):
            chosen = candidate & ~assigned & (selector < cumulative[k])
            selected.append(chosen)
            assigned = assigned | chosen
        elastic = selected[0]
        levels = selected[1:-1]
        ionization = selected[-1]
        excitation = levels[0]
        for chosen in levels[1:]:
            excitation = excitation | chosen
        null = candidate & ~(elastic | excitation | ionization)
        thresholds = self.table.thresholds_ev
        # Elastic: isotropic, same speed.
        if np.any(elastic):
            nvx, nvy, nvz = isotropic_velocity(speed[elastic], u[2][elastic], u[3][elastic])
            vx[elastic], vy[elastic], vz[elastic] = nvx, nvy, nvz
        # Excitation: lose the level's threshold, isotropic (v2.3.0: per level; one level = the legacy lumped loss).
        for k, chosen in enumerate(levels):
            if np.any(chosen):
                remaining = np.maximum(energy[chosen] - thresholds[1 + k], 0.0)
                nvx, nvy, nvz = isotropic_velocity(electron_speed_from_energy(remaining), u[2][chosen], u[3][chosen])
                vx[chosen], vy[chosen], vz[chosen] = nvx, nvy, nvz
        new_electrons = ParticleArrays.empty()
        new_ions = ParticleArrays.empty()
        ion_r = np.zeros(0)
        ion_z = np.zeros(0)
        if np.any(ionization):
            available = np.maximum(energy[ionization] - thresholds[-1], 0.0)
            secondary = VAHEDI_SURENDRA_B_EV * np.tan(u[4][ionization] * np.arctan(available / (2.0 * VAHEDI_SURENDRA_B_EV)))
            secondary = np.clip(secondary, 0.0, available)
            primary = available - secondary
            pvx, pvy, pvz = isotropic_velocity(electron_speed_from_energy(primary), u[2][ionization], u[3][ionization])
            vx[ionization], vy[ionization], vz[ionization] = pvx, pvy, pvz
            svx, svy, svz = isotropic_velocity(electron_speed_from_energy(secondary), u[5][ionization], u[6][ionization])
            ion_r = electrons.r_m[ionization].copy()
            ion_z = electrons.z_m[ionization].copy()
            new_electrons = ParticleArrays(ion_r.copy(), ion_z.copy(), svx, svy, svz)
            ivx, ivy, ivz = maxwellian_velocity(
                self.ion_species.mass_kg, self.config.neutral_temperature_k, u[7:11][:, ionization]
            )
            new_ions = ParticleArrays(ion_r.copy(), ion_z.copy(), ivx, ivy, ivz)
        level_counts = tuple(int(np.count_nonzero(chosen)) for chosen in levels)
        n_exc = sum(level_counts)
        n_ion = int(np.count_nonzero(ionization))
        # per macro event; the ledger applies W (v2.0.6).  v2.3.0: sum over the levels of count x threshold (for the
        # one-level legacy set this is the v2.0.6 expression ``(n_exc E_exc + n_ion E_ion) e`` with the same arithmetic)
        loss_ev = 0.0
        for k, count in enumerate(level_counts):
            loss_ev += count * thresholds[1 + k]
        loss = (loss_ev + n_ion * thresholds[-1]) * EV_J
        tally = MCCTally(n_candidates, int(np.count_nonzero(elastic)), n_exc, n_ion, int(np.count_nonzero(null)), loss, level_counts)
        updated = ParticleArrays(electrons.r_m.copy(), electrons.z_m.copy(), vx, vy, vz)
        return MCCResult(updated, new_electrons, new_ions, ion_r, ion_z, tally)

    def to_dict(self) -> dict[str, Any]:
        levels = self.cross_sections.excitation_levels
        if len(levels) == 1:
            excitation = "lumped 8.32 eV loss, isotropic"
        else:
            excitation = "per-level threshold loss (" + ", ".join(f"{p.identifier} {p.threshold_ev} eV" for p in levels) + "), isotropic (v2.3.0)"
        return {
            "config": self.config.to_dict(),
            "cross_sections": self.cross_sections.to_dict(),
            "uniform_table_sha256": self.table.sha256(),
            "nu_max_per_s": self.nu_max,
            "kinematics": {
                "elastic": "isotropic, speed preserved (2 m_e/M loss neglected)",
                "excitation": excitation,
                "ionization": "12.13 eV loss; Vahedi-Surendra secondary energy with B=8.7 eV; both electrons isotropic; ion born with Maxwellian neutral velocity",
                "energy_definition": "classical 0.5 m_e v^2 in the lab frame (neutral at rest)",
            },
        }


def _expm1(value: float) -> float:
    if abs(value) < 1.0e-5:
        return value + value * value / 2.0 + value**3 / 6.0 + value**4 / 24.0
    return exp(value) - 1.0


__all__ = [
    "CROSS_SECTION_SCHEMAS",
    "DEFAULT_CROSS_SECTION_PATH",
    "MAX_EXCITATION_LEVELS",
    "CrossSectionProcess",
    "MCCConfig",
    "MCCResult",
    "MCCTally",
    "NullCollisionMCC",
    "PROCESS_ORDER",
    "UniformSigmaTable",
    "VAHEDI_SURENDRA_B_EV",
    "XenonCrossSections",
    "canonical_payload_sha256",
    "electron_energy_ev",
    "electron_speed_from_energy",
    "isotropic_velocity",
    "maximum_collision_frequency",
    "maxwellian_velocity",
    "validate_process_order",
]
