from __future__ import annotations

import pytest

from cft_revival.pic import (
    Grid1D,
    PICDeviceError,
    PICValidationError,
    ParticleState,
    Species,
    cic_deposit_charge,
    deposit_and_push_warp,
    device_available,
    integrated_charge_c,
    push_electrostatic_leapfrog,
    represented_charge_c,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_face_gather_push_and_cic_match_python(device: str) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} unavailable")
    grid = Grid1D(-0.5, 0.5, 16, transverse_area_m2=0.25)
    species = Species("parity", charge_c=-2.0, mass_kg=5.0, macro_weight=0.25)
    original = ParticleState(
        [-0.5, -0.499, -0.27, 0.0, 0.19, 0.499],
        [0.2, -0.1, 0.0, 0.4, -0.3, 0.1],
        [1.0] * 6,
        [-1.0] * 6,
    )
    face_field = [
        2.0 + 0.5 * index - 0.03 * index * index
        for index in range(grid.cells)
    ]
    expected_density = cic_deposit_charge(grid, species, original)
    expected_particles = original.copy()
    push_electrostatic_leapfrog(
        grid, species, expected_particles, face_field, 0.01
    )
    actual = deposit_and_push_warp(
        grid, species, original, face_field, 0.01, device=device
    )
    assert actual.deposited_charge_density_c_per_m3 == pytest.approx(
        expected_density, rel=2.0e-15, abs=1.0e-14
    )
    assert actual.particles.x_m == pytest.approx(
        expected_particles.x_m, rel=2.0e-15, abs=2.0e-16
    )
    assert actual.particles.vx_m_per_s == pytest.approx(
        expected_particles.vx_m_per_s, rel=2.0e-15, abs=2.0e-16
    )
    assert actual.particles.vy_m_per_s == original.vy_m_per_s
    assert actual.particles.vz_m_per_s == original.vz_m_per_s


def test_invalid_warp_device_is_typed() -> None:
    grid = Grid1D(0.0, 1.0, 4)
    species = Species("unit", 1.0, 1.0)
    particles = ParticleState([0.5], [0.0], [0.0], [0.0])
    with pytest.raises(PICDeviceError):
        deposit_and_push_warp(
            grid, species, particles, [0.0] * grid.cells, 0.1, device="gpu"
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("charge_c", [1.0e-8, 1.0e-9])
def test_warp_extreme_area_preserves_every_accepted_charge(
    device: str, charge_c: float
) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} unavailable")
    grid = Grid1D(0.0, 1.0, 8, transverse_area_m2=1.0e300)
    species = Species("boundary", charge_c, 1.0)
    particles = ParticleState([0.1, 0.9], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    result = deposit_and_push_warp(
        grid,
        species,
        particles,
        [0.0] * grid.cells,
        0.01,
        device=device,
    )
    assert any(value != 0.0 for value in result.deposited_charge_density_c_per_m3)
    assert integrated_charge_c(
        grid, result.deposited_charge_density_c_per_m3
    ) == pytest.approx(
        represented_charge_c(species, particles.count),
        rel=1.5e-14,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_rejects_unrepresentable_extreme_area_before_launch(device: str) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} unavailable")
    grid = Grid1D(0.0, 1.0, 8, transverse_area_m2=1.0e300)
    species = Species("underflow", 1.0e-300, 1.0)
    particles = ParticleState([0.25], [0.0], [0.0], [0.0])
    with pytest.raises(PICValidationError, match="volumetric particle charge density"):
        deposit_and_push_warp(
            grid,
            species,
            particles,
            [0.0] * grid.cells,
            0.01,
            device=device,
        )
