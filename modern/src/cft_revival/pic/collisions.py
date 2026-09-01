"""Deterministic Monte Carlo collision operator with auditable cross sections."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import acos, cos, exp, hypot, isfinite, pi, sin
from random import Random

from .models import ELEMENTARY_CHARGE_C, PICValidationError, ParticleState, Species


@dataclass(frozen=True, slots=True)
class CrossSectionTable:
    """Piecewise-linear energy/cross-section data.

    Production LXCat imports must populate ``source`` and
    ``external_data_sha256`` with the byte hash of the downloaded source.
    Synthetic tables are permitted only when their source starts with
    ``synthetic-verification:``.
    """

    process: str
    energy_ev: tuple[float, ...]
    cross_section_m2: tuple[float, ...]
    source: str
    external_data_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.process.strip() or not self.source.strip():
            raise PICValidationError("cross-section process and source are required")
        if len(self.energy_ev) < 2 or len(self.energy_ev) != len(self.cross_section_m2):
            raise PICValidationError("cross-section arrays must have equal length >= 2")
        energies = tuple(float(value) for value in self.energy_ev)
        sections = tuple(float(value) for value in self.cross_section_m2)
        if any(not isfinite(value) for value in (*energies, *sections)):
            raise PICValidationError("cross-section data must be finite")
        if energies[0] < 0.0 or any(b <= a for a, b in zip(energies, energies[1:])):
            raise PICValidationError("cross-section energies must strictly increase")
        if any(value < 0.0 for value in sections):
            raise PICValidationError("cross sections must be non-negative")
        if self.source.startswith("synthetic-verification:"):
            if self.external_data_sha256 is not None:
                raise PICValidationError("synthetic data must not claim an external hash")
        elif (
            self.external_data_sha256 is None
            or len(self.external_data_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.external_data_sha256)
        ):
            raise PICValidationError("external cross sections require a lowercase SHA-256")
        object.__setattr__(self, "energy_ev", energies)
        object.__setattr__(self, "cross_section_m2", sections)

    @property
    def table_sha256(self) -> str:
        payload = json.dumps(
            {
                "process": self.process,
                "energy_ev": self.energy_ev,
                "cross_section_m2": self.cross_section_m2,
                "source": self.source,
                "external_data_sha256": self.external_data_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    def at_energy_ev(self, energy_ev: float) -> float:
        energy = float(energy_ev)
        if not isfinite(energy) or energy < 0.0:
            raise PICValidationError("collision energy must be finite and non-negative")
        if energy <= self.energy_ev[0]:
            return self.cross_section_m2[0]
        if energy >= self.energy_ev[-1]:
            return self.cross_section_m2[-1]
        for index in range(len(self.energy_ev) - 1):
            lower = self.energy_ev[index]
            upper = self.energy_ev[index + 1]
            if lower <= energy <= upper:
                fraction = (energy - lower) / (upper - lower)
                return (
                    (1.0 - fraction) * self.cross_section_m2[index]
                    + fraction * self.cross_section_m2[index + 1]
                )
        raise AssertionError("validated interpolation interval was not found")


@dataclass(frozen=True, slots=True)
class MCCDiagnostics:
    trials: int
    accepted_collisions: int
    expected_collisions: float
    maximum_probability: float
    seed: int
    cross_section_sha256: str


class ElasticMCC:
    """Seeded electron/neutral elastic-scattering operator.

    The target is stationary and infinitely massive. Accepted events randomize
    direction isotropically and preserve particle speed exactly apart from
    floating-point roundoff.
    """

    def __init__(
        self,
        table: CrossSectionTable,
        target_density_per_m3: float,
        seed: int,
        *,
        max_probability: float = 0.2,
    ) -> None:
        density = float(target_density_per_m3)
        probability_limit = float(max_probability)
        if not isfinite(density) or density < 0.0:
            raise PICValidationError("target density must be finite and non-negative")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or not isfinite(probability_limit)
            or not 0.0 < probability_limit < 1.0
        ):
            raise PICValidationError("seed and MCC probability limit are invalid")
        self.table = table
        self.target_density_per_m3 = density
        self.seed = seed
        self.max_probability = probability_limit
        self.rng = Random(seed)
        self.trial_count = 0
        self.accepted_count = 0

    def apply(
        self, species: Species, particles: ParticleState, dt_s: float
    ) -> MCCDiagnostics:
        dt = float(dt_s)
        if not isfinite(dt) or dt <= 0.0:
            raise PICValidationError("collision dt_s must be finite and positive")
        particles.validate()
        probabilities: list[float] = []
        speeds: list[float] = []
        for index in range(particles.count):
            vx = particles.vx_m_per_s[index]
            vy = particles.vy_m_per_s[index]
            vz = particles.vz_m_per_s[index]
            speed = hypot(vx, vy, vz)
            energy_ev = 0.5 * species.mass_kg * speed * speed / ELEMENTARY_CHARGE_C
            if not isfinite(speed) or not isfinite(energy_ev):
                raise PICValidationError(
                    f"particle {index} collision energy is not representable"
                )
            frequency = (
                self.target_density_per_m3
                * self.table.at_energy_ev(energy_ev)
                * speed
            )
            if not isfinite(frequency):
                raise PICValidationError(
                    f"particle {index} collision frequency is not representable"
                )
            probability = -expm1_safe(-frequency * dt)
            if not isfinite(probability) or not 0.0 <= probability <= self.max_probability:
                raise PICValidationError(
                    f"particle {index} MCC probability {probability!r} violates "
                    f"configured limit {self.max_probability:.6g}"
                )
            speeds.append(speed)
            probabilities.append(probability)

        proposed_vx = particles.vx_m_per_s.copy()
        proposed_vy = particles.vy_m_per_s.copy()
        proposed_vz = particles.vz_m_per_s.copy()
        proposed_rng = Random()
        proposed_rng.setstate(self.rng.getstate())
        accepted = 0
        for index, (speed, probability) in enumerate(
            zip(speeds, probabilities, strict=True)
        ):
            if proposed_rng.random() >= probability or speed == 0.0:
                continue
            cosine = 2.0 * proposed_rng.random() - 1.0
            azimuth = 2.0 * pi * proposed_rng.random()
            sine = sin(acos(cosine))
            vx = speed * sine * cos(azimuth)
            vy = speed * sine * sin(azimuth)
            vz = speed * cosine
            if not isfinite(vx) or not isfinite(vy) or not isfinite(vz):
                raise PICValidationError(
                    f"particle {index} MCC proposal is not representable"
                )
            proposed_vx[index] = vx
            proposed_vy[index] = vy
            proposed_vz[index] = vz
            accepted += 1
        proposed_trials = self.trial_count + particles.count
        proposed_accepted = self.accepted_count + accepted
        if proposed_trials < self.trial_count or proposed_accepted < self.accepted_count:
            raise PICValidationError("MCC counters are not representable")

        particles.vx_m_per_s[:] = proposed_vx
        particles.vy_m_per_s[:] = proposed_vy
        particles.vz_m_per_s[:] = proposed_vz
        self.rng.setstate(proposed_rng.getstate())
        self.trial_count = proposed_trials
        self.accepted_count = proposed_accepted
        return MCCDiagnostics(
            particles.count,
            accepted,
            sum(probabilities),
            max(probabilities, default=0.0),
            self.seed,
            self.table.table_sha256,
        )


def expm1_safe(value: float) -> float:
    """Return ``exp(value)-1`` accurately enough for small negative rates."""

    if abs(value) < 1.0e-5:
        # Terms through x^4 bound the verification regime well below binary64 noise.
        return value + value * value / 2.0 + value**3 / 6.0 + value**4 / 24.0
    return exp(value) - 1.0
