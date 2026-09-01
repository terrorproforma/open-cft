"""Integrated, deliberately small CPU electrostatic PIC-MCC reference step."""

from __future__ import annotations

from ast import literal_eval
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import metadata, util
import json
from math import fsum, hypot, isclose, isfinite
from pathlib import Path
import platform
import sys
from typing import Any

from .collisions import ElasticMCC, MCCDiagnostics
from .electrostatic import (
    ElectrostaticField,
    PeriodicPoisson1D,
    PoissonSolver,
    add_uniform_background,
    cic_deposit_charge,
    integrated_charge_c,
    physical_number_density_per_m3,
    represented_charge_c,
)
from .models import (
    EPSILON_0_F_PER_M,
    Grid1D,
    PICConfig,
    PICConvergenceError,
    PICError,
    PICValidationError,
    ParticleState,
    PoissonConfig,
    Species,
)
from .particles import push_electrostatic_leapfrog

PIC_CODE_REVISION = "pic-foundation-r2-staggered-transactional"


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    step: int
    time_s: float
    total_particle_charge_c: float
    deposited_charge_c: float
    kinetic_energy_j: float
    field_energy_j: float
    total_energy_j: float
    energy_stagger: str
    poisson_relative_residual: float
    collisions: MCCDiagnostics | None


def kinetic_energy_j(species: Species, particles: ParticleState) -> float:
    """Return actual kinetic energy represented in the named grid cross-section."""

    particles.validate()
    factor = 0.5 * species.mass_kg * species.macro_weight
    if not isfinite(factor):
        raise PICValidationError("kinetic-energy scale is not representable")
    terms: list[float] = []
    for vx, vy, vz in zip(
            particles.vx_m_per_s,
            particles.vy_m_per_s,
            particles.vz_m_per_s,
            strict=True,
    ):
        speed = hypot(vx, vy, vz)
        term = factor * speed * speed
        if not isfinite(term):
            raise PICValidationError("particle kinetic energy is not representable")
        terms.append(term)
    energy = fsum(terms)
    if not isfinite(energy):
        raise PICValidationError("total kinetic energy is not representable")
    return energy


def field_energy_j(grid: Grid1D, field: ElectrostaticField) -> float:
    """Return field energy in joules over ``length * transverse_area_m2``."""

    energy = (
        0.5
        * EPSILON_0_F_PER_M
        * grid.transverse_area_m2
        * grid.dx_m
        * fsum(value * value for value in field.electric_field_face_v_per_m)
    )
    if not isfinite(energy):
        raise PICValidationError("electrostatic field energy is not representable")
    return energy


def field_energy_j_per_m2(grid: Grid1D, field: ElectrostaticField) -> float:
    """Compatibility helper returning energy divided by the explicit area."""

    return field_energy_j(grid, field) / grid.transverse_area_m2


class PICStepper:
    """Reduced CPU integrator with transactional state publication."""

    def __init__(
        self,
        grid: Grid1D,
        species: Species,
        particles: ParticleState,
        config: PICConfig,
        *,
        poisson_solver: PoissonSolver[Grid1D] | None = None,
        collision_operator: ElasticMCC | None = None,
    ) -> None:
        if any(not grid.x_min_m <= x < grid.x_max_m for x in particles.x_m):
            raise PICValidationError("initial particles must lie inside the periodic grid")
        self.grid = grid
        self.species = species
        self.particles = particles
        self.config = config
        self.poisson_solver = poisson_solver or PeriodicPoisson1D()
        self.collision_operator = collision_operator
        self.step_index = 0
        self.last_field: ElectrostaticField | None = None

    def _physical_density(self) -> float:
        return physical_number_density_per_m3(
            self.grid, self.species, self.particles.count
        )

    def _enforce_stability(self, particles: ParticleState, *, stage: str) -> None:
        from .models import stability_report

        report = stability_report(
            self.grid,
            self.species,
            particles,
            self.config,
            self._physical_density(),
        )
        if not report.stable:
            raise PICValidationError(
                f"{stage} stability gate failed: {', '.join(report.violations)}"
            )

    def _solve_field_for(self, particles: ParticleState) -> ElectrostaticField:
        particles.validate()
        deposited = cic_deposit_charge(self.grid, self.species, particles)
        net_density = add_uniform_background(
            deposited, self.config.background_charge_density_c_per_m3
        )
        try:
            field = self.poisson_solver.solve(
                self.grid, net_density, self.config.poisson
            )
        except PICError:
            raise
        except (ArithmeticError, ValueError, TypeError) as error:
            raise PICConvergenceError("Poisson solver failed with an invalid state") from error
        if (
            len(field.potential_v) != self.grid.cells
            or len(field.electric_field_face_v_per_m) != self.grid.cells
        ):
            raise PICConvergenceError("Poisson solver returned an invalid field shape")
        diagnostics = field.diagnostics
        numeric = (
            *field.potential_v,
            *field.electric_field_face_v_per_m,
            field.removed_mean_charge_density_c_per_m3,
            diagnostics.initial_residual_l2,
            diagnostics.final_residual_l2,
            diagnostics.required_residual_l2,
        )
        try:
            finite_field = all(isfinite(float(value)) for value in numeric)
        except (TypeError, ValueError, OverflowError) as error:
            raise PICConvergenceError(
                "Poisson solver returned nonnumeric field diagnostics"
            ) from error
        if not finite_field:
            raise PICConvergenceError("Poisson solver returned nonfinite field diagnostics")
        if (
            not isinstance(diagnostics.converged, bool)
            or isinstance(diagnostics.iterations, bool)
            or not isinstance(diagnostics.iterations, int)
            or diagnostics.iterations < 0
        ):
            raise PICConvergenceError("Poisson solver returned malformed diagnostics")
        if not diagnostics.converged:
            raise PICConvergenceError("Poisson solver returned an unconverged field")
        return field

    def solve_field(self) -> ElectrostaticField:
        field = self._solve_field_for(self.particles)
        self.last_field = field
        return field

    def step(self) -> StepDiagnostics:
        self.particles.validate()
        self._enforce_stability(self.particles, stage="pre-step")
        field_before = self._solve_field_for(self.particles)
        deposited_before_push = cic_deposit_charge(self.grid, self.species, self.particles)
        working = self.particles.copy()
        try:
            represented_charge = represented_charge_c(self.species, working.count)
            deposited_charge = integrated_charge_c(
                self.grid, deposited_before_push
            )
            next_step = self.step_index + 1
            next_time = next_step * self.config.dt_s
        except ArithmeticError as error:
            raise PICValidationError(
                "pre-step charge/time diagnostics are not representable"
            ) from error
        if any(
            not isfinite(value)
            for value in (represented_charge, deposited_charge, next_time)
        ):
            raise PICValidationError("pre-step charge/time diagnostics are not finite")
        if not isclose(
            represented_charge,
            deposited_charge,
            rel_tol=32.0 * sys.float_info.epsilon,
            abs_tol=0.0,
        ):
            raise PICValidationError("deposited charge does not conserve represented charge")
        collision_snapshot = None
        if self.collision_operator is not None:
            collision_snapshot = (
                self.collision_operator.rng.getstate(),
                self.collision_operator.trial_count,
                self.collision_operator.accepted_count,
            )
        try:
            push_electrostatic_leapfrog(
                self.grid,
                self.species,
                working,
                field_before.electric_field_face_v_per_m,
                self.config.dt_s,
            )
            working.validate()
            self._enforce_stability(working, stage="post-push")
            collision_diagnostics = (
                None
                if self.collision_operator is None
                else self.collision_operator.apply(
                    self.species, working, self.config.dt_s
                )
            )
            working.validate()
            self._enforce_stability(working, stage="post-collision")
            field_after = self._solve_field_for(working)
            kinetic = kinetic_energy_j(self.species, working)
            field_energy = 0.5 * (
                field_energy_j(self.grid, field_before)
                + field_energy_j(self.grid, field_after)
            )
            total_energy = kinetic + field_energy
            initial = field_after.diagnostics.initial_residual_l2
            relative = (
                0.0
                if initial == 0.0
                else field_after.diagnostics.final_residual_l2 / initial
            )
            if any(
                not isfinite(value)
                for value in (
                    kinetic,
                    field_energy,
                    total_energy,
                    relative,
                )
            ):
                raise PICValidationError("PIC diagnostics are not finite")
            result = StepDiagnostics(
                next_step,
                next_time,
                represented_charge,
                deposited_charge,
                kinetic,
                field_energy,
                total_energy,
                "K(v[n+1/2]) + 0.5*(UE[x[n]] + UE[x[n+1]])",
                relative,
                collision_diagnostics,
            )
        except PICError:
            if self.collision_operator is not None and collision_snapshot is not None:
                rng_state, trials, accepted = collision_snapshot
                self.collision_operator.rng.setstate(rng_state)
                self.collision_operator.trial_count = trials
                self.collision_operator.accepted_count = accepted
            raise
        except (ArithmeticError, ValueError) as error:
            if self.collision_operator is not None and collision_snapshot is not None:
                rng_state, trials, accepted = collision_snapshot
                self.collision_operator.rng.setstate(rng_state)
                self.collision_operator.trial_count = trials
                self.collision_operator.accepted_count = accepted
            raise PICValidationError("PIC step produced an invalid numerical state") from error

        self.particles.x_m[:] = working.x_m
        self.particles.vx_m_per_s[:] = working.vx_m_per_s
        self.particles.vy_m_per_s[:] = working.vy_m_per_s
        self.particles.vz_m_per_s[:] = working.vz_m_per_s
        self.last_field = field_after
        self.step_index = next_step
        return result

    def checkpoint(self) -> dict[str, Any]:
        collision = self.collision_operator
        payload: dict[str, Any] = {
            "schema": "cft.pic.checkpoint.v2",
            "code_revision": PIC_CODE_REVISION,
            "backend": "python",
            "device": "cpu",
            "runtime": _runtime_record(),
            "step": self.step_index,
            "time_s": self.step_index * self.config.dt_s,
            "staggering": {
                "positions": "integer-step",
                "velocities": "half-step-behind-position",
                "field": "not-stored; recompute faces at position-step",
                "energy": "K_half + mean(UE_integer_before,UE_integer_after)",
            },
            "grid": asdict(self.grid),
            "species": asdict(self.species),
            "config": {
                **asdict(self.config),
                "poisson": asdict(self.config.poisson),
            },
            "particles": asdict(self.particles),
            "collision": None,
        }
        if collision is not None:
            payload["collision"] = {
                "seed": collision.seed,
                "target_density_per_m3": collision.target_density_per_m3,
                "max_probability": collision.max_probability,
                "cross_section_sha256": collision.table.table_sha256,
                "rng_state": repr(collision.rng.getstate()),
                "trial_count": collision.trial_count,
                "accepted_count": collision.accepted_count,
            }
        payload["payload_sha256"] = _payload_hash(payload)
        return payload

    def restore_collision_rng(self, checkpoint: dict[str, Any]) -> None:
        """Restore MCC state only from a closed, identity-matched checkpoint."""

        validate_checkpoint(
            checkpoint,
            expected_grid=self.grid,
            expected_species=self.species,
            expected_config=self.config,
        )
        if self.collision_operator is None or checkpoint.get("collision") is None:
            raise PICValidationError("checkpoint and stepper must both contain MCC state")
        collision = checkpoint["collision"]
        operator = self.collision_operator
        if collision["cross_section_sha256"] != operator.table.table_sha256:
            raise PICValidationError("checkpoint cross-section hash does not match")
        if (
            collision["seed"] != operator.seed
            or collision["target_density_per_m3"] != operator.target_density_per_m3
            or collision["max_probability"] != operator.max_probability
        ):
            raise PICValidationError("checkpoint MCC configuration does not match")
        operator.rng.setstate(literal_eval(collision["rng_state"]))
        operator.trial_count = collision["trial_count"]
        operator.accepted_count = collision["accepted_count"]


def write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    validate_checkpoint(checkpoint)
    destination = Path(path)
    destination.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _payload_hash(payload: dict[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("payload_sha256", None)
    try:
        canonical = json.dumps(
            unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise PICValidationError("checkpoint is not finite canonical JSON") from error
    return sha256(canonical.encode("utf-8")).hexdigest()


def _closed(mapping: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(mapping, dict) or set(mapping) != expected:
        raise PICValidationError(f"{context} must have exactly {sorted(expected)}")
    return mapping


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PICValidationError(f"{context} must be numeric")
    converted = float(value)
    if not isfinite(converted):
        raise PICValidationError(f"{context} must be finite")
    return converted


def _runtime_record() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "byteorder": sys.byteorder,
        "float_format": f"binary64-mantissa-{sys.float_info.mant_dig}",
    }


def validate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_grid: Grid1D | None = None,
    expected_species: Species | None = None,
    expected_config: PICConfig | None = None,
) -> None:
    top = _closed(
        checkpoint,
        {
            "schema",
            "code_revision",
            "backend",
            "device",
            "runtime",
            "step",
            "time_s",
            "staggering",
            "grid",
            "species",
            "config",
            "particles",
            "collision",
            "payload_sha256",
        },
        "checkpoint",
    )
    if top["schema"] != "cft.pic.checkpoint.v2":
        raise PICValidationError("unsupported PIC checkpoint schema")
    if top["code_revision"] != PIC_CODE_REVISION:
        raise PICValidationError("checkpoint code revision does not match")
    if top["backend"] != "python" or top["device"] != "cpu":
        raise PICValidationError("checkpoint backend/device is unsupported")
    supplied_hash = checkpoint.get("payload_sha256")
    if (
        not isinstance(supplied_hash, str)
        or len(supplied_hash) != 64
        or any(character not in "0123456789abcdef" for character in supplied_hash)
    ):
        raise PICValidationError("checkpoint payload SHA-256 is missing")
    if _payload_hash(checkpoint) != supplied_hash:
        raise PICValidationError("checkpoint payload SHA-256 does not match")

    runtime = _closed(
        top["runtime"],
        {
            "python_implementation",
            "python_version",
            "python_compiler",
            "platform",
            "machine",
            "byteorder",
            "float_format",
        },
        "checkpoint runtime",
    )
    if any(not isinstance(value, str) or not value for value in runtime.values()):
        raise PICValidationError("checkpoint runtime values must be nonempty strings")
    if runtime["byteorder"] not in {"little", "big"}:
        raise PICValidationError("checkpoint byteorder is invalid")
    if runtime != _runtime_record():
        raise PICValidationError("checkpoint runtime identity does not match")
    staggering = _closed(
        top["staggering"],
        {"positions", "velocities", "field", "energy"},
        "checkpoint staggering",
    )
    required_staggering = {
        "positions": "integer-step",
        "velocities": "half-step-behind-position",
        "field": "not-stored; recompute faces at position-step",
        "energy": "K_half + mean(UE_integer_before,UE_integer_after)",
    }
    if staggering != required_staggering:
        raise PICValidationError("checkpoint staggering contract does not match")

    grid_values = _closed(
        top["grid"],
        {"x_min_m", "x_max_m", "cells", "transverse_area_m2", "geometry"},
        "checkpoint grid",
    )
    species_values = _closed(
        top["species"],
        {"name", "charge_c", "mass_kg", "macro_weight"},
        "checkpoint species",
    )
    config_values = _closed(
        top["config"],
        {
            "dt_s",
            "background_charge_density_c_per_m3",
            "seed",
            "poisson",
            "max_particle_courant",
            "max_omega_p_dt",
        },
        "checkpoint config",
    )
    poisson_values = _closed(
        config_values["poisson"],
        {"relative_tolerance", "absolute_tolerance", "max_iterations"},
        "checkpoint Poisson config",
    )
    particles_values = _closed(
        top["particles"],
        {"x_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s"},
        "checkpoint particles",
    )
    for key in ("x_min_m", "x_max_m", "transverse_area_m2"):
        _finite_number(grid_values[key], f"checkpoint grid.{key}")
    for key in ("charge_c", "mass_kg", "macro_weight"):
        _finite_number(species_values[key], f"checkpoint species.{key}")
    for key in (
        "dt_s",
        "background_charge_density_c_per_m3",
        "max_particle_courant",
        "max_omega_p_dt",
    ):
        _finite_number(config_values[key], f"checkpoint config.{key}")
    for key in ("relative_tolerance", "absolute_tolerance"):
        _finite_number(poisson_values[key], f"checkpoint config.poisson.{key}")
    if not isinstance(grid_values["geometry"], str):
        raise PICValidationError("checkpoint grid.geometry must be a string")
    if not isinstance(species_values["name"], str):
        raise PICValidationError("checkpoint species.name must be a string")
    for key, values in particles_values.items():
        if not isinstance(values, list) or not values:
            raise PICValidationError(f"checkpoint particles.{key} must be a nonempty list")
        for index, value in enumerate(values):
            _finite_number(value, f"checkpoint particles.{key}[{index}]")

    try:
        grid = Grid1D(**grid_values)
        species = Species(**species_values)
        poisson = PoissonConfig(**poisson_values)
        config_payload = dict(config_values)
        config_payload["poisson"] = poisson
        config = PICConfig(**config_payload)
        particles = ParticleState(**particles_values)
    except (TypeError, PICError) as error:
        raise PICValidationError("checkpoint typed state is invalid") from error
    if any(not grid.x_min_m <= x < grid.x_max_m for x in particles.x_m):
        raise PICValidationError("checkpoint particles lie outside the periodic grid")
    if expected_grid is not None and grid != expected_grid:
        raise PICValidationError("checkpoint grid identity does not match")
    if expected_species is not None and species != expected_species:
        raise PICValidationError("checkpoint species identity does not match")
    if expected_config is not None and config != expected_config:
        raise PICValidationError("checkpoint config identity does not match")

    step = top["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise PICValidationError("checkpoint step must be a non-negative integer")
    time_s = _finite_number(top["time_s"], "checkpoint time_s")
    if time_s != step * config.dt_s:
        raise PICValidationError("checkpoint time is inconsistent with step and dt")

    collision = top["collision"]
    if collision is not None:
        values = _closed(
            collision,
            {
                "seed",
                "target_density_per_m3",
                "max_probability",
                "cross_section_sha256",
                "rng_state",
                "trial_count",
                "accepted_count",
            },
            "checkpoint collision",
        )
        if (
            isinstance(values["seed"], bool)
            or not isinstance(values["seed"], int)
            or values["seed"] < 0
        ):
            raise PICValidationError("checkpoint collision seed is invalid")
        target_density = _finite_number(
            values["target_density_per_m3"],
            "checkpoint collision target density",
        )
        if target_density < 0.0:
            raise PICValidationError("checkpoint collision target density is negative")
        probability_limit = _finite_number(
            values["max_probability"], "checkpoint collision probability limit"
        )
        if not 0.0 < probability_limit < 1.0:
            raise PICValidationError("checkpoint collision probability limit is invalid")
        digest = values["cross_section_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PICValidationError("checkpoint cross-section SHA-256 is invalid")
        trials = values["trial_count"]
        accepted = values["accepted_count"]
        if (
            isinstance(trials, bool)
            or isinstance(accepted, bool)
            or not isinstance(trials, int)
            or not isinstance(accepted, int)
            or trials < 0
            or accepted < 0
            or accepted > trials
        ):
            raise PICValidationError("checkpoint MCC counters are inconsistent")
        if not isinstance(values["rng_state"], str):
            raise PICValidationError("checkpoint RNG state must be a string")
        try:
            temporary_rng = __import__("random").Random()
            temporary_rng.setstate(literal_eval(values["rng_state"]))
        except (ValueError, TypeError, SyntaxError) as error:
            raise PICValidationError("checkpoint RNG state is invalid") from error


def provenance_record(*, cross_section_sha256: str | None = None) -> dict[str, Any]:
    """Capture runtime and data identity without claiming external validation."""

    if cross_section_sha256 is not None and (
        len(cross_section_sha256) != 64
        or any(character not in "0123456789abcdef" for character in cross_section_sha256)
    ):
        raise PICValidationError("provenance cross-section hash must be lowercase SHA-256")

    def available(module: str) -> bool:
        try:
            return util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    def version(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    record: dict[str, Any] = {
        "schema": "cft.pic.provenance.v2",
        "code_revision": PIC_CODE_REVISION,
        "backend": "python",
        "device": "cpu",
        "runtime": _runtime_record(),
        "solver": "periodic-node-potential-face-field-fd-cg/python-binary64",
        "staggering": {
            "charge_and_potential": "nodes",
            "electric_field": "faces",
            "positions": "integer-step",
            "velocities": "half-step-behind-position",
        },
        "optional_dependencies": {
            "warp": {
                "available": available("warp"),
                "version": version("warp-lang"),
            },
            "warpx_picmi": {
                "available": available("pywarpx"),
                "version": version("pywarpx"),
                "verified": False,
            },
            "amrex": {
                "available": available("amrex"),
                "version": version("amrex"),
                "verified": False,
            },
        },
        "cross_section_sha256": cross_section_sha256,
        "claim": "reduced-kernel-verification-only",
    }
    record["record_sha256"] = _payload_hash(record)
    return record
