"""Scrambled Sobol': net properties, determinism, extension, reference points."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.orbit_wall_loss_geometry_screening_v2 import sobol as S

SEEDS = (1, 2, 7, 12345678901234, 2**64 - 1)


def test_unscrambled_direction_numbers_reproduce_the_standard_sobol_points(monkeypatch: pytest.MonkeyPatch) -> None:
    identity_rows = [1 << (31 - i) for i in range(32)]

    class NoRandom:
        def integers(self, lo, hi, size=None, dtype=None):  # noqa: ANN001
            return np.uint64(0) if size is None else np.zeros(size, dtype=np.uint64)

    monkeypatch.setattr(S, "_scramble_rows", lambda rng: identity_rows)
    monkeypatch.setattr(S.np.random, "default_rng", lambda seed: NoRandom())
    points = S.scrambled_sobol(3, 8, 1)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.375, 0.375, 0.625],
            [0.875, 0.875, 0.125],
            [0.625, 0.125, 0.875],
            [0.125, 0.625, 0.375],
        ]
    )
    assert np.array_equal(points, expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_prefixes_are_balanced_nets_and_the_first_two_dimensions_form_a_0_m_2_net(seed: int) -> None:
    points = S.scrambled_sobol(3, 64, seed)
    assert points.shape == (64, 3)
    assert np.all((points >= 0.0) & (points < 1.0))
    for bits in range(1, 7):
        assert S.dyadic_balance(points, bits)
    for m in (16, 32, 64):
        prefix = points[:m]
        assert S.dyadic_balance(prefix, int(np.log2(m)))
    q = points[:16, :2]
    for a, b in ((2, 2), (1, 3), (3, 1), (0, 4), (4, 0)):
        boxes = np.floor(q[:, 0] * 2**a).astype(int) * 2**b + np.floor(q[:, 1] * 2**b).astype(int)
        assert np.all(np.bincount(boxes, minlength=16) == 1)
    boxes = np.floor(points[:, 0] * 8).astype(int) * 8 + np.floor(points[:, 1] * 8).astype(int)
    assert np.all(np.bincount(boxes, minlength=64) == 1)


@pytest.mark.parametrize("seed", SEEDS)
def test_sequence_is_deterministic_and_stage_two_extends_stage_one(seed: int) -> None:
    full = S.scrambled_sobol(3, 64, seed)
    assert np.array_equal(full, S.scrambled_sobol(3, 64, seed))
    assert np.array_equal(full[:16], S.scrambled_sobol(3, 16, seed))
    assert np.array_equal(full[16:], S.scrambled_sobol(3, 48, seed, start=16))
    assert np.array_equal(full[16:32], S.scrambled_sobol(3, 16, seed, start=16))
    assert np.array_equal(full[48:64], S.scrambled_sobol(3, 16, seed, start=48))


def test_seeds_and_dimensions_differ_and_first_dimension_halves_split_exactly() -> None:
    a = S.scrambled_sobol(3, 16, 7)
    b = S.scrambled_sobol(3, 16, 8)
    assert not np.array_equal(a, b)
    assert len({tuple(row) for row in a}) == 16
    for seed in SEEDS:
        band = S.scrambled_sobol(3, 16, seed)[:, 0] < 0.5
        assert band.sum() == 8


def test_seed_from_bytes_is_uint64_and_stable() -> None:
    seed = S.seed_from_bytes(b"ns:design:cell:E0:P0:D+1")
    assert 0 <= seed < 2**64
    assert seed == S.seed_from_bytes(b"ns:design:cell:E0:P0:D+1")
    assert seed != S.seed_from_bytes(b"ns:design:cell:E0:P0:D-1")


def test_invalid_arguments_are_refused() -> None:
    with pytest.raises(ValueError):
        S.scrambled_sobol(0, 4, 1)
    with pytest.raises(ValueError):
        S.scrambled_sobol(S.MAX_DIMENSION + 1, 4, 1)
    with pytest.raises(ValueError):
        S.scrambled_sobol(2, -1, 1)
    with pytest.raises(ValueError):
        S.scrambled_sobol(2, 4, 2**64)
    with pytest.raises(ValueError):
        S.scrambled_sobol(2, 4, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        S.dyadic_balance(np.zeros((6, 1)), 2)
