from math import atan, cos, fsum, sin

import pytest

from cft_revival.hybrid import (
    ELEMENTARY_CHARGE_C,
    AxisAlignedBox,
    BoundaryPolicy,
    CartesianGrid1D,
    HybridValidationError,
    Particle,
    UniformFields,
    VelocityTimeLevel,
    XE,
    XE_DOUBLE_PLUS,
    XE_PLUS,
    XenonSpecies,
    advance_particles,
    apply_boundary,
    boris_push,
    boris_push_diagnosed,
    deposit_cic_periodic,
    initialize_leapfrog,
    particle_kinetic_energy,
    particle_momentum,
    run_tiny_manufactured_case,
    synchronize_velocity,
)


def test_uniform_e_acceleration_and_work_energy_identity() -> None:
    particle = Particle(0, XE_PLUS, (0.0, 0.0, 0.0), (7.0, -2.0, 1.0), weight=3.0)
    fields = UniformFields(electric_v_per_m=(25.0, 0.0, 0.0))
    dt = 2.0e-8

    advanced = boris_push(particle, fields, dt)
    acceleration = ELEMENTARY_CHARGE_C * 25.0 / XE_PLUS.mass_kg
    expected_vx = particle.velocity_m_per_s[0] + acceleration * dt
    assert advanced.velocity_m_per_s == pytest.approx((expected_vx, -2.0, 1.0))
    assert advanced.position_m == pytest.approx((expected_vx * dt, -2.0 * dt, dt))

    energy_delta = particle_kinetic_energy(advanced) - particle_kinetic_energy(particle)
    electric_work = (
        particle.represented_charge_c
        * 25.0
        * 0.5
        * (particle.velocity_m_per_s[0] + advanced.velocity_m_per_s[0])
        * dt
    )
    assert energy_delta == pytest.approx(electric_work, rel=2.0e-14, abs=1.0e-38)


def test_leapfrog_constant_e_analytic_displacement_and_work() -> None:
    fields = UniformFields(electric_v_per_m=(25.0, 0.0, 0.0))
    dt = 2.0e-8
    step_count = 40
    synchronous_initial = Particle(
        50,
        XE_PLUS,
        (0.0, 0.0, 0.0),
        (7.0, 0.0, 0.0),
        weight=2.0,
        velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
    )
    state = initialize_leapfrog(synchronous_initial, fields, dt)
    half_initial_energy = particle_kinetic_energy(state)
    step_work = 0.0
    for _ in range(step_count):
        result = boris_push_diagnosed(state, fields, dt)
        step_work += result.electric_work_j
        assert result.work_energy_residual_j == pytest.approx(
            0.0, abs=5.0e-38
        )
        state = result.particle

    elapsed = step_count * dt
    acceleration = XE_PLUS.charge_c * 25.0 / XE_PLUS.mass_kg
    expected_x = (
        synchronous_initial.position_m[0]
        + synchronous_initial.velocity_m_per_s[0] * elapsed
        + 0.5 * acceleration * elapsed * elapsed
    )
    assert state.position_m[0] == pytest.approx(expected_x, rel=2.0e-15)
    assert (
        particle_kinetic_energy(state) - half_initial_energy
        == pytest.approx(step_work, rel=2.0e-14, abs=2.0e-38)
    )

    synchronous_final = synchronize_velocity(state, fields, dt)
    assert synchronous_final.velocity_m_per_s[0] == pytest.approx(
        synchronous_initial.velocity_m_per_s[0] + acceleration * elapsed
    )
    physical_energy_delta = (
        particle_kinetic_energy(synchronous_final)
        - particle_kinetic_energy(synchronous_initial)
    )
    physical_work = (
        synchronous_initial.represented_charge_c
        * 25.0
        * (state.position_m[0] - synchronous_initial.position_m[0])
    )
    assert physical_energy_delta == pytest.approx(
        physical_work, rel=2.0e-14, abs=2.0e-38
    )


def test_boris_gyro_rotation_and_energy_are_correct() -> None:
    magnetic_t = 0.03
    gyrofrequency = XE_PLUS.charge_c * magnetic_t / XE_PLUS.mass_kg
    dt = 0.2 / gyrofrequency
    particle = Particle(1, XE_PLUS, (0.0, 0.0, 0.0), (400.0, 0.0, 0.0))

    advanced = boris_push(particle, UniformFields(magnetic_t=(0.0, 0.0, magnetic_t)), dt)
    angle = 2.0 * atan(0.5 * gyrofrequency * dt)
    assert advanced.velocity_m_per_s == pytest.approx(
        (400.0 * cos(angle), -400.0 * sin(angle), 0.0),
        rel=2.0e-15,
        abs=1.0e-14,
    )
    assert particle_kinetic_energy(advanced) == pytest.approx(
        particle_kinetic_energy(particle), rel=3.0e-16
    )

    state = particle
    for _ in range(200):
        state = boris_push(
            state, UniformFields(magnetic_t=(0.0, 0.0, magnetic_t)), dt
        )
    assert particle_kinetic_energy(state) == pytest.approx(
        particle_kinetic_energy(particle), rel=2.0e-14
    )


def test_long_leapfrog_gyro_is_bounded_and_nondissipative() -> None:
    magnetic_t = 0.03
    fields = UniformFields(magnetic_t=(0.0, 0.0, magnetic_t))
    gyrofrequency = XE_PLUS.charge_c * magnetic_t / XE_PLUS.mass_kg
    dt = 0.02 / gyrofrequency
    synchronous = Particle(
        51,
        XE_PLUS,
        (0.0, 0.0, 0.0),
        (400.0, 0.0, 0.0),
        velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
    )
    state = initialize_leapfrog(synchronous, fields, dt)
    initial_energy = particle_kinetic_energy(state)
    maximum_radius = 0.0
    for _ in range(5000):
        state = boris_push(state, fields, dt)
        maximum_radius = max(
            maximum_radius,
            (state.position_m[0] ** 2 + state.position_m[1] ** 2) ** 0.5,
        )
    assert particle_kinetic_energy(state) == pytest.approx(
        initial_energy, rel=3.0e-13
    )
    assert maximum_radius <= 2.01 * 400.0 / gyrofrequency


def test_zero_step_and_zero_field_limits() -> None:
    particle = Particle(3, XE_DOUBLE_PLUS, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    assert boris_push(particle, UniformFields(electric_v_per_m=(9.0, 8.0, 7.0)), 0.0) is particle
    drifted = boris_push(particle, UniformFields(), 0.25)
    assert drifted.velocity_m_per_s == particle.velocity_m_per_s
    assert drifted.position_m == pytest.approx((2.0, 3.25, 4.5))


@pytest.mark.parametrize(
    ("policy", "expected_x", "expected_vx", "alive"),
    [
        (BoundaryPolicy.PERIODIC, 0.2, 3.0, True),
        (BoundaryPolicy.REFLECTING, 0.8, -3.0, True),
        (BoundaryPolicy.ABSORBING, 1.2, 3.0, False),
    ],
)
def test_boundary_policies_and_exchange(
    policy: BoundaryPolicy, expected_x: float, expected_vx: float, alive: bool
) -> None:
    particle = Particle(4, XE_PLUS, (1.2, 0.5, 0.5), (3.0, 0.0, 0.0))
    result = apply_boundary(
        particle,
        AxisAlignedBox((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), policy),
    )
    assert result.particle.position_m[0] == pytest.approx(expected_x)
    assert result.particle.velocity_m_per_s[0] == pytest.approx(expected_vx)
    assert result.particle.alive is alive
    assert result.source_exchange.momentum_residual_kg_m_per_s == pytest.approx(
        (0.0, 0.0, 0.0), abs=0.0
    )
    assert result.source_exchange.energy_residual_j == 0.0


def test_cic_deposition_conserves_integrated_moments() -> None:
    particles = (
        Particle(0, XE, (0.0, 0.0, 0.0), (2.0, 3.0, 4.0), weight=2.0),
        Particle(1, XE_PLUS, (0.31, 0.0, 0.0), (-1.0, 5.0, 2.0), weight=3.0),
        Particle(2, XE_DOUBLE_PLUS, (1.0, 0.0, 0.0), (7.0, -2.0, 1.0)),
    )
    grid = CartesianGrid1D(0.0, 1.0, 8, transverse_area_m2=0.4)
    moments = deposit_cic_periodic(particles, grid)
    volume = grid.cell_volume_m3

    assert fsum(moments.number_per_m3) * volume == pytest.approx(6.0)
    assert fsum(moments.charge_c_per_m3) * volume == pytest.approx(
        5.0 * ELEMENTARY_CHARGE_C
    )
    expected_current = tuple(
        fsum(
            particle.represented_charge_c * particle.velocity_m_per_s[axis]
            for particle in particles
        )
        for axis in range(3)
    )
    expected_momentum = tuple(
        fsum(particle_momentum(particle)[axis] for particle in particles)
        for axis in range(3)
    )
    for axis in range(3):
        assert fsum(row[axis] for row in moments.current_a_per_m2) * volume == pytest.approx(
            expected_current[axis], rel=2.0e-15, abs=1.0e-40
        )
        assert (
            fsum(row[axis] for row in moments.momentum_kg_per_m2_s) * volume
            == pytest.approx(expected_momentum[axis], rel=2.0e-15, abs=1.0e-40)
        )
    assert fsum(moments.kinetic_energy_j_per_m3) * volume == pytest.approx(
        fsum(particle_kinetic_energy(particle) for particle in particles),
        rel=2.0e-15,
    )


def test_tiny_manufactured_run_is_a_consistent_fixture() -> None:
    result = run_tiny_manufactured_case()
    assert len(result.particles) == 3
    assert result.work_energy_residual_j == pytest.approx(0.0, abs=2.0e-37)
    assert result.electrons.electric_field_v_per_m is None
    assert result.electrons.state.anomalous_mobility_m2_per_v_s is None
    assert "verification fixture" in result.claim


def test_reference_failures_are_typed() -> None:
    particle = Particle(0, XE_PLUS, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(HybridValidationError, match="non-negative"):
        boris_push(particle, UniformFields(), -1.0)
    with pytest.raises(HybridValidationError, match="outside"):
        deposit_cic_periodic(
            (Particle(1, XE, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),),
            CartesianGrid1D(0.0, 1.0, 4),
        )
    with pytest.raises(HybridValidationError):
        advance_particles((object(),), UniformFields(), 0.0)  # type: ignore[arg-type]
    with pytest.raises(HybridValidationError, match="unsigned 64-bit"):
        Particle(1 << 64, XE, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(HybridValidationError, match="non-negative"):
        run_tiny_manufactured_case(step_count=0, dt_s=-1.0)
    synchronous = Particle(
        8,
        XE_PLUS,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        velocity_time_level=VelocityTimeLevel.SYNCHRONOUS_N,
    )
    with pytest.raises(HybridValidationError, match="velocity_time_level"):
        boris_push(synchronous, UniformFields(), 1.0)
    with pytest.raises(HybridValidationError, match="boolean"):
        Particle(
            9,
            XE,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            alive=1,  # type: ignore[arg-type]
        )


def test_duplicate_ids_fail_before_deposition_or_advance() -> None:
    duplicate = (
        Particle(90, XE_PLUS, (0.2, 0.0, 0.0), (1.0, 0.0, 0.0)),
        Particle(90, XE_PLUS, (0.3, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    with pytest.raises(HybridValidationError, match="unique"):
        deposit_cic_periodic(duplicate, CartesianGrid1D(0.0, 1.0, 4))
    with pytest.raises(HybridValidationError, match="unique"):
        advance_particles(duplicate, UniformFields(), 1.0e-8)


def test_numeric_models_reject_coercible_non_real_objects() -> None:
    class FloatLike:
        def __float__(self) -> float:
            return 1.0

    with pytest.raises(HybridValidationError, match="real finite"):
        Particle(
            91,
            XE_PLUS,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            weight=FloatLike(),  # type: ignore[arg-type]
        )
    with pytest.raises(HybridValidationError, match="real finite"):
        Particle(
            92,
            XE_PLUS,
            (FloatLike(), 0.0, 0.0),  # type: ignore[arg-type]
            (0.0, 0.0, 0.0),
        )
    for invalid_id in (True, 1.0):
        with pytest.raises(HybridValidationError, match="unsigned 64-bit"):
            Particle(
                invalid_id,  # type: ignore[arg-type]
                XE_PLUS,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )


@pytest.mark.parametrize(
    ("symbol", "charge_state", "bad_charge"),
    [
        ("Xe", 0, ELEMENTARY_CHARGE_C),
        ("Xe+", 1, 0.0),
        ("Xe+", 1, 2.0 * ELEMENTARY_CHARGE_C),
        ("Xe2+", 2, ELEMENTARY_CHARGE_C),
    ],
)
def test_xenon_charge_is_derived_from_integer_charge_state(
    symbol: str, charge_state: int, bad_charge: float
) -> None:
    with pytest.raises(HybridValidationError, match="elementary charge"):
        XenonSpecies(
            symbol,
            charge_state,
            mass_kg=2.25e-25,
            identifier=f"{symbol}-custom-mass",
            charge_c_override=bad_charge,
        )
    valid = XenonSpecies(
        symbol,
        charge_state,
        mass_kg=2.25e-25,
        identifier=f"{symbol}-valid-custom-mass",
    )
    assert valid.mass_kg == 2.25e-25
    assert valid.charge_c == charge_state * ELEMENTARY_CHARGE_C
