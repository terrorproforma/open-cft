from __future__ import annotations

from math import cos, fsum, pi, sin

import pytest

from cft_revival.pic import (
    EPSILON_0_F_PER_M,
    Grid1D,
    PICConvergenceError,
    PICValidationError,
    ParticleState,
    PeriodicPoisson1D,
    PoissonConfig,
    Species,
    boris_push_uniform,
    cic_deposit_charge,
    field_energy_j,
    gather_face_cic,
    gather_cic,
    integrated_charge_c,
    represented_charge_c,
    push_electrostatic_leapfrog,
)


def test_cic_deposition_conserves_charge_and_is_adjoint_to_gather() -> None:
    grid = Grid1D(-0.5, 0.5, 32)
    species = Species("test ion", 2.5e-12, 3.0, macro_weight=7.0)
    particles = ParticleState(
        [-0.5, -0.499, -0.13, 0.0, 0.499999],
        [0.0] * 5,
        [0.0] * 5,
        [0.0] * 5,
    )
    density = cic_deposit_charge(grid, species, particles)
    represented = species.charge_c * species.macro_weight * particles.count
    assert fsum(density) * grid.dx_m == pytest.approx(represented, rel=2.0e-16)

    node_field = tuple(sin(2.0 * pi * i / grid.cells) for i in range(grid.cells))
    particle_field = gather_cic(grid, node_field, particles.x_m)
    grid_pairing = fsum(
        rho * field for rho, field in zip(density, node_field, strict=True)
    ) * grid.dx_m
    particle_pairing = species.charge_c * species.macro_weight * fsum(particle_field)
    assert grid_pairing == pytest.approx(particle_pairing, rel=2.0e-15)


@pytest.mark.parametrize("charge_c", [1.0e-8, 1.0e-9])
def test_extreme_area_accepted_cpu_deposition_preserves_charge(charge_c: float) -> None:
    grid = Grid1D(0.0, 1.0, 8, transverse_area_m2=1.0e300)
    species = Species("trace", charge_c, 1.0)
    particles = ParticleState([0.1, 0.9], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    density = cic_deposit_charge(grid, species, particles)
    assert any(value != 0.0 for value in density)
    assert integrated_charge_c(grid, density) == pytest.approx(
        represented_charge_c(species, particles.count),
        rel=5.0e-15,
    )


def test_extreme_area_unrepresentable_density_is_rejected_not_zeroed() -> None:
    grid = Grid1D(0.0, 1.0, 8, transverse_area_m2=1.0e300)
    species = Species("underflow", 1.0e-300, 1.0)
    particles = ParticleState([0.25], [0.0], [0.0], [0.0])
    with pytest.raises(PICValidationError, match="volumetric particle charge density"):
        cic_deposit_charge(grid, species, particles)


def test_manufactured_periodic_poisson_mode_and_field() -> None:
    grid = Grid1D(0.0, 2.0, 64)
    mode = 3
    wave_number = 2.0 * pi * mode / grid.length_m
    discrete_eigenvalue = (
        4.0
        * sin(pi * mode / grid.cells) ** 2
        / (grid.dx_m * grid.dx_m)
    )
    expected_phi = tuple(
        cos(2.0 * pi * mode * i / grid.cells) for i in range(grid.cells)
    )
    rho = tuple(
        EPSILON_0_F_PER_M * discrete_eigenvalue * value
        for value in expected_phi
    )
    result = PeriodicPoisson1D().solve(grid, rho, PoissonConfig(1.0e-12, 1.0e-11))
    assert result.diagnostics.converged
    assert max(
        abs(actual - expected)
        for actual, expected in zip(result.potential_v, expected_phi, strict=True)
    ) < 2.0e-14
    expected_field_scale = 2.0 * sin(wave_number * grid.dx_m / 2.0) / grid.dx_m
    for index, actual in enumerate(result.electric_field_face_v_per_m):
        expected = expected_field_scale * sin(
            2.0 * pi * mode * (index + 0.5) / grid.cells
        )
        assert actual == pytest.approx(expected, abs=2.0e-13)


def test_nyquist_mode_has_nonzero_face_field_and_poisson_energy_identity() -> None:
    grid = Grid1D(0.0, 1.0, 16, transverse_area_m2=3.0)
    expected_phi = tuple(1.0 if index % 2 == 0 else -1.0 for index in range(grid.cells))
    eigenvalue = 4.0 / (grid.dx_m * grid.dx_m)
    rho = tuple(EPSILON_0_F_PER_M * eigenvalue * value for value in expected_phi)
    field = PeriodicPoisson1D().solve(grid, rho)
    assert min(abs(value) for value in field.electric_field_face_v_per_m) > 0.0
    source_energy = (
        0.5
        * grid.transverse_area_m2
        * grid.dx_m
        * fsum(
            charge * potential
            for charge, potential in zip(rho, field.potential_v, strict=True)
        )
    )
    assert field_energy_j(grid, field) == pytest.approx(source_energy, rel=3.0e-15)


@pytest.mark.parametrize("position", [0.01, 0.1, 0.49, 0.99])
def test_periodic_single_particle_has_no_resolved_self_force(position: float) -> None:
    grid = Grid1D(0.0, 1.0, 32)
    species = Species("test charge", 1.0e-12, 1.0)
    particles = ParticleState([position], [0.0], [0.0], [0.0])
    density = cic_deposit_charge(grid, species, particles)
    neutral = tuple(value - species.charge_c / grid.length_m for value in density)
    field = PeriodicPoisson1D().solve(grid, neutral)
    force_field = gather_face_cic(
        grid, field.electric_field_face_v_per_m, particles.x_m
    )[0]
    assert force_field == pytest.approx(0.0, abs=3.0e-16)


def test_extreme_poisson_sources_never_converge_through_infinity() -> None:
    grid = Grid1D(0.0, 1.0, 8)
    unrepresentable = tuple(1.0e298 if i % 2 == 0 else -1.0e298 for i in range(8))
    with pytest.raises(PICValidationError, match="not representable"):
        PeriodicPoisson1D().solve(grid, unrepresentable)

    finite_entries_unrepresentable_norm = tuple(
        1.0e297 if i % 2 == 0 else -1.0e297 for i in range(8)
    )
    with pytest.raises(PICConvergenceError, match="L2 norm"):
        PeriodicPoisson1D().solve(grid, finite_entries_unrepresentable_norm)

    near_limit = tuple(1.0e296 if i % 2 == 0 else -1.0e296 for i in range(8))
    result = PeriodicPoisson1D().solve(grid, near_limit)
    assert result.diagnostics.converged
    assert all(
        value < float("inf")
        for value in (
            result.diagnostics.initial_residual_l2,
            result.diagnostics.final_residual_l2,
            result.diagnostics.required_residual_l2,
            *result.potential_v,
            *result.electric_field_face_v_per_m,
        )
    )


def test_overflowed_tolerance_cannot_make_inf_less_equal_inf_converge() -> None:
    grid = Grid1D(0.0, 1.0, 8)
    rho = tuple(
        EPSILON_0_F_PER_M * sin(2.0 * pi * index / grid.cells)
        for index in range(grid.cells)
    )
    with pytest.raises(PICValidationError, match="tolerance"):
        PeriodicPoisson1D().solve(
            grid,
            rho,
            PoissonConfig(relative_tolerance=1.0e308),
        )


def test_poisson_nonconvergence_is_explicit_or_inspectable() -> None:
    grid = Grid1D(0.0, 1.0, 32)
    rho = tuple(
        EPSILON_0_F_PER_M
        * (
            sin(2.0 * pi * index / grid.cells)
            + 0.4 * cos(6.0 * pi * index / grid.cells)
        )
        for index in range(grid.cells)
    )
    config = PoissonConfig(relative_tolerance=1.0e-14, max_iterations=1)
    with pytest.raises(PICConvergenceError, match="did not converge"):
        PeriodicPoisson1D().solve(grid, rho, config)
    result = PeriodicPoisson1D().solve(
        grid, rho, config, raise_on_nonconvergence=False
    )
    assert not result.diagnostics.converged
    assert result.diagnostics.final_residual_l2 > result.diagnostics.required_residual_l2


def test_single_particle_constant_field_acceleration_and_periodic_drift() -> None:
    grid = Grid1D(0.0, 1.0, 16)
    species = Species("unit", charge_c=2.0, mass_kg=4.0)
    particles = ParticleState([0.99], [0.1], [0.0], [0.0])
    push_electrostatic_leapfrog(
        grid, species, particles, [3.0] * grid.cells, 0.2
    )
    assert particles.vx_m_per_s[0] == pytest.approx(0.4)
    assert particles.x_m[0] == pytest.approx(0.07)


def test_nonrepresentable_push_is_typed_and_all_or_nothing() -> None:
    grid = Grid1D(0.0, 1.0, 8)
    species = Species("extreme", charge_c=1.0e308, mass_kg=1.0)
    particles = ParticleState([0.25, 0.75], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    before = particles.copy()
    with pytest.raises(PICValidationError, match="acceleration scale"):
        push_electrostatic_leapfrog(
            grid,
            species,
            particles,
            [1.0] * grid.cells,
            2.0,
        )
    assert particles == before


def test_boris_magnetic_rotation_preserves_speed() -> None:
    species = Species("unit", charge_c=1.0, mass_kg=2.0)
    particles = ParticleState([0.0], [3.0], [4.0], [5.0])
    before = fsum(value * value for value in (3.0, 4.0, 5.0))
    boris_push_uniform(species, particles, (0.0, 0.0, 0.0), (0.2, -0.1, 2.0), 0.3)
    after = fsum(
        value * value
        for value in (
            particles.vx_m_per_s[0],
            particles.vy_m_per_s[0],
            particles.vz_m_per_s[0],
        )
    )
    assert after == pytest.approx(before, rel=2.0e-16)
