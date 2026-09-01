from __future__ import annotations

import json
from hashlib import sha256
from math import fsum, pi, sin, sqrt

import pytest

from cft_revival.pic import (
    CrossSectionTable,
    EPSILON_0_F_PER_M,
    ElasticMCC,
    ElectrostaticField,
    Grid1D,
    PICConfig,
    PICConvergenceError,
    PICStepper,
    PICValidationError,
    ParticleState,
    PoissonDiagnostics,
    PoissonConfig,
    Species,
    provenance_record,
    stability_report,
    validate_checkpoint,
    write_checkpoint,
)


def _plasma(dt_s: float) -> PICStepper:
    count = 64
    grid = Grid1D(0.0, 1.0, count)
    charge = -sqrt(EPSILON_0_F_PER_M)
    species = Species("normalized electron", charge, 1.0, macro_weight=1.0 / count)
    unperturbed = [(index + 0.5) / count for index in range(count)]
    particles = ParticleState(
        [
            (position + 0.01 * sin(2.0 * pi * position)) % 1.0
            for position in unperturbed
        ],
        [0.0] * count,
        [0.0] * count,
        [0.0] * count,
    )
    return PICStepper(
        grid,
        species,
        particles,
        PICConfig(
            dt_s,
            background_charge_density_c_per_m3=-charge,
            poisson=PoissonConfig(relative_tolerance=1.0e-10, absolute_tolerance=1.0e-8),
        ),
    )


def test_cold_plasma_mode_has_expected_quarter_period_and_conserves_deposition() -> None:
    stepper = _plasma(0.02)
    amplitudes: list[float] = []
    diagnostics = []
    for _ in range(100):
        result = stepper.step()
        diagnostics.append(result)
        field = stepper.last_field
        assert field is not None
        amplitudes.append(
            fsum(
                value * sin(2.0 * pi * (index + 0.5) / stepper.grid.cells)
                for index, value in enumerate(field.electric_field_face_v_per_m)
            )
        )
    crossing = next(
        index
        for index in range(1, len(amplitudes))
        if amplitudes[index - 1] * amplitudes[index] <= 0.0
    )
    # This normalization gives omega_p = 1 rad/s, hence T/4 = pi/2.
    assert (crossing + 1) * stepper.config.dt_s == pytest.approx(pi / 2.0, rel=0.03)
    assert all(
        item.deposited_charge_c
        == pytest.approx(item.total_particle_charge_c, rel=2.0e-15)
        for item in diagnostics
    )
    assert max(item.poisson_relative_residual for item in diagnostics) < 1.0e-10


def test_smaller_timestep_reduces_cold_plasma_energy_envelope() -> None:
    def relative_energy_envelope(dt_s: float) -> float:
        stepper = _plasma(dt_s)
        energies = [
            stepper.step().total_energy_j
            for _ in range(round(4.0 / dt_s))
        ]
        return (max(energies) - min(energies)) / energies[0]

    fine = relative_energy_envelope(0.02)
    coarse = relative_energy_envelope(0.1)
    assert fine < 0.03
    assert coarse > 1.5 * fine


def test_stability_report_exposes_cell_and_plasma_frequency_limits() -> None:
    grid = Grid1D(0.0, 1.0, 10)
    species = Species("unit", sqrt(EPSILON_0_F_PER_M), 1.0)
    particles = ParticleState([0.1], [2.0], [0.0], [0.0])
    report = stability_report(
        grid,
        species,
        particles,
        PICConfig(0.1, max_particle_courant=1.0, max_omega_p_dt=0.05),
        physical_number_density_per_m3=1.0,
    )
    assert not report.stable
    assert report.particle_courant == pytest.approx(2.0)
    assert report.omega_p_dt == pytest.approx(0.1)
    assert len(report.violations) == 2


@pytest.mark.parametrize(
    "damage",
    [
        lambda particles, species: particles.x_m.__setitem__(0, float("nan")),
        lambda particles, species: particles.vx_m_per_s.__setitem__(0, float("nan")),
        lambda particles, species: particles.vy_m_per_s.__setitem__(0, float("nan")),
        lambda particles, species: particles.vz_m_per_s.__setitem__(0, float("nan")),
        lambda particles, species: object.__setattr__(
            species, "macro_weight", float("nan")
        ),
        lambda particles, species: particles.vy_m_per_s.append(0.0),
    ],
)
def test_stability_zero_density_never_bypasses_complete_state_validation(damage) -> None:
    grid = Grid1D(0.0, 1.0, 8)
    species = Species("unit", 1.0e-12, 1.0)
    particles = ParticleState([0.25], [0.0], [0.0], [0.0])
    damage(particles, species)
    with pytest.raises(PICValidationError):
        stability_report(grid, species, particles, PICConfig(0.01), 0.0)


def test_stability_metrics_are_finite_for_valid_zero_density_state() -> None:
    report = stability_report(
        Grid1D(0.0, 1.0, 8),
        Species("unit", 1.0e-12, 1.0),
        ParticleState([0.25], [0.5], [2.0], [3.0]),
        PICConfig(0.01),
        0.0,
    )
    assert report.stable
    assert report.particle_courant == pytest.approx(0.04)
    assert report.omega_p_dt == 0.0


def test_extreme_area_step_diagnostics_preserve_accepted_charge() -> None:
    grid = Grid1D(0.0, 1.0, 8, transverse_area_m2=1.0e300)
    species = Species("trace", 1.0e-9, 1.0)
    particles = ParticleState([0.25, 0.75], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    result = PICStepper(
        grid,
        species,
        particles,
        PICConfig(0.01, background_charge_density_c_per_m3=-2.0e-309),
    ).step()
    assert result.total_particle_charge_c == pytest.approx(2.0e-9, rel=3.0e-15)
    assert result.deposited_charge_c == pytest.approx(
        result.total_particle_charge_c, rel=3.0e-15
    )


def test_checkpoint_hash_round_trip_and_provenance_are_auditable(tmp_path) -> None:
    stepper = _plasma(0.02)
    stepper.step()
    checkpoint = stepper.checkpoint()
    validate_checkpoint(checkpoint)
    destination = tmp_path / "pic-checkpoint.json"
    write_checkpoint(destination, checkpoint)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    validate_checkpoint(loaded)
    assert destination.read_text(encoding="utf-8").endswith("\n")
    loaded["step"] += 1
    with pytest.raises(PICValidationError, match="does not match"):
        validate_checkpoint(loaded)

    provenance = provenance_record()
    assert provenance["claim"] == "reduced-kernel-verification-only"
    assert provenance["optional_dependencies"]["warpx_picmi"]["verified"] is False
    assert provenance["optional_dependencies"]["amrex"]["verified"] is False
    assert provenance["staggering"]["electric_field"] == "faces"
    assert len(provenance["record_sha256"]) == 64


def _rehash(checkpoint: dict) -> None:
    unhashed = dict(checkpoint)
    unhashed.pop("payload_sha256", None)
    canonical = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    checkpoint["payload_sha256"] = sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize(
    "damage",
    [
        lambda value: value.update({"unexpected": 1}),
        lambda value: value["particles"].update({"x_m": []}),
        lambda value: value["particles"]["vx_m_per_s"].pop(),
        lambda value: value.update({"code_revision": "tampered"}),
        lambda value: value.update({"backend": "cuda"}),
        lambda value: value["staggering"].update({"field": "nodes"}),
        lambda value: value.update({"time_s": value["time_s"] + 0.5}),
        lambda value: value["runtime"].update({"machine": ""}),
    ],
)
def test_rehashed_malformed_checkpoints_are_rejected(damage) -> None:
    stepper = _plasma(0.02)
    checkpoint = stepper.checkpoint()
    damage(checkpoint)
    _rehash(checkpoint)
    with pytest.raises(PICValidationError):
        validate_checkpoint(checkpoint)


def test_checkpoint_identity_mismatch_is_rejected_even_when_hash_is_valid() -> None:
    stepper = _plasma(0.02)
    checkpoint = stepper.checkpoint()
    other_species = Species("other", stepper.species.charge_c, 1.0, 1.0 / 64)
    with pytest.raises(PICValidationError, match="species identity"):
        validate_checkpoint(checkpoint, expected_species=other_species)


def test_step_rejects_unstable_state_before_particle_mutation() -> None:
    grid = Grid1D(0.0, 1.0, 8)
    species = Species("unit", sqrt(EPSILON_0_F_PER_M), 1.0)
    particles = ParticleState([0.25], [100.0], [0.0], [0.0])
    before = particles.copy()
    stepper = PICStepper(grid, species, particles, PICConfig(0.1))
    with pytest.raises(PICValidationError, match="pre-step stability"):
        stepper.step()
    assert particles == before
    assert stepper.step_index == 0
    assert stepper.last_field is None


def test_post_push_stability_failure_is_transactional() -> None:
    class StrongFiniteField:
        def solve(self, grid, charge_density, config, *, raise_on_nonconvergence=True):
            diagnostics = PoissonDiagnostics(True, 0, 0.0, 0.0, config.absolute_tolerance)
            return ElectrostaticField(
                (0.0,) * grid.cells,
                (1.0e6,) * grid.cells,
                diagnostics,
                0.0,
            )

    grid = Grid1D(0.0, 1.0, 8)
    species = Species("scaled", 1.0e-12, 1.0e-12)
    particles = ParticleState([0.25], [0.0], [0.0], [0.0])
    before = particles.copy()
    stepper = PICStepper(
        grid,
        species,
        particles,
        PICConfig(0.1),
        poisson_solver=StrongFiniteField(),
    )
    with pytest.raises(PICValidationError, match="post-push stability"):
        stepper.step()
    assert particles == before
    assert stepper.step_index == 0
    assert stepper.last_field is None


def test_nonfinite_injected_field_is_rejected_before_mutation() -> None:
    class NonfiniteField:
        def solve(self, grid, charge_density, config, *, raise_on_nonconvergence=True):
            diagnostics = PoissonDiagnostics(True, 0, 0.0, 0.0, config.absolute_tolerance)
            return ElectrostaticField(
                (0.0,) * grid.cells,
                (0.0,) * (grid.cells - 1) + (float("nan"),),
                diagnostics,
                0.0,
            )

    grid = Grid1D(0.0, 1.0, 8)
    species = Species("scaled", 1.0e-12, 1.0)
    particles = ParticleState([0.25], [0.0], [0.0], [0.0])
    before = particles.copy()
    stepper = PICStepper(
        grid,
        species,
        particles,
        PICConfig(0.01),
        poisson_solver=NonfiniteField(),
    )
    with pytest.raises(PICConvergenceError, match="nonfinite"):
        stepper.step()
    assert particles == before
    assert stepper.step_index == 0


def test_externally_mutated_nonfinite_state_raises_typed_error() -> None:
    stepper = _plasma(0.02)
    stepper.particles.vx_m_per_s[-1] = float("nan")
    with pytest.raises(PICValidationError, match="finite"):
        stepper.step()
    assert stepper.step_index == 0


def test_step_collision_failure_rolls_back_particles_rng_and_counters() -> None:
    grid = Grid1D(0.0, 1.0, 8)
    species = Species("unit", 1.0e-19, 1.0)
    particles = ParticleState([0.25, 0.75], [0.1, 0.1], [0.0, 0.0], [0.0, 0.0])
    table = CrossSectionTable(
        "synthetic",
        (0.0, 1.0),
        (1.0, 1.0),
        "synthetic-verification:rollback",
    )
    operator = ElasticMCC(table, 1.0e6, seed=7)
    stepper = PICStepper(
        grid,
        species,
        particles,
        PICConfig(0.01, background_charge_density_c_per_m3=-2.0e-19),
        collision_operator=operator,
    )
    before = particles.copy()
    rng_before = operator.rng.getstate()
    with pytest.raises(PICValidationError, match="MCC probability"):
        stepper.step()
    assert particles == before
    assert operator.rng.getstate() == rng_before
    assert (operator.trial_count, operator.accepted_count) == (0, 0)
    assert stepper.step_index == 0
    assert stepper.last_field is None


@pytest.mark.parametrize(
    "damage",
    [
        lambda collision: collision.update({"trial_count": -1}),
        lambda collision: collision.update({"accepted_count": 1, "trial_count": 0}),
        lambda collision: collision.update({"rng_state": "not a Random state"}),
        lambda collision: collision.update({"cross_section_sha256": "0" * 63}),
    ],
)
def test_rehashed_malformed_collision_checkpoint_is_rejected(damage) -> None:
    grid = Grid1D(0.0, 1.0, 8)
    species = Species("unit", 1.0e-19, 1.0)
    particles = ParticleState([0.25], [0.0], [0.0], [0.0])
    table = CrossSectionTable(
        "synthetic",
        (0.0, 1.0),
        (1.0e-20, 1.0e-20),
        "synthetic-verification:checkpoint",
    )
    operator = ElasticMCC(table, 1.0, seed=7)
    checkpoint = PICStepper(
        grid,
        species,
        particles,
        PICConfig(0.01),
        collision_operator=operator,
    ).checkpoint()
    damage(checkpoint["collision"])
    _rehash(checkpoint)
    with pytest.raises(PICValidationError):
        validate_checkpoint(checkpoint)
