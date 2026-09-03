"""Scrambled Sobol sampler: definition checks, net balance, determinism."""

from __future__ import annotations

import pytest

from cft_revival.optimization import Variable

from experiments.l1a_geometry_sweep_v3 import sampling as S


def test_unscrambled_first_dimensions_are_the_textbook_sequences() -> None:
    points = S.sobol_points(8, 2, seed=0, scramble=False)
    assert [p[0] for p in points] == [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125]
    assert [p[1] for p in points] == [0.0, 0.5, 0.25, 0.75, 0.375, 0.875, 0.125, 0.625]
    assert S.direction_numbers(1)[:3] == [1 << 31, 1 << 30, 1 << 29]
    assert S.direction_numbers(3)[:2] == [1 << 31, 3 << 30]


@pytest.mark.parametrize("seed", [0, 20260903, 7])
def test_scrambled_points_keep_the_net_balance(seed: int) -> None:
    points = S.sobol_points(128, 11, seed=seed)
    assert len(points) == 128 and all(len(p) == 11 and all(0.0 <= c < 1.0 for c in p) for p in points)
    for dimension in range(11):
        bins = [0] * 128
        for point in points:
            bins[int(point[dimension] * 128)] += 1
        assert all(count == 1 for count in bins), dimension
    # Two-dimensional balance holds for the projections whose t-value is 0 (the first
    # dimensions); higher pairs of a Sobol net have t > 0 and are not required to be balanced.
    for a, b in ((0, 1), (1, 2), (0, 2)):
        boxes: dict[tuple[int, int], int] = {}
        for point in points:
            key = (int(point[a] * 8), int(point[b] * 16))
            boxes[key] = boxes.get(key, 0) + 1
        assert len(boxes) == 128 and max(boxes.values()) == 1, (a, b)


def test_scramble_is_deterministic_and_seed_dependent() -> None:
    assert S.sobol_points(16, 4, seed=1) == S.sobol_points(16, 4, seed=1)
    assert S.sobol_points(16, 4, seed=1) != S.sobol_points(16, 4, seed=2)
    assert S.sobol_points(16, 4, seed=1) != S.sobol_points(16, 4, seed=1, scramble=False)


def test_sobol_designs_scale_to_the_box_and_are_unique() -> None:
    variables = (Variable("a", 0.0, 1.0, "1"), Variable("b", 2.0, 5.0, "m"), Variable("c", -1.0, 1.0, "1"))
    designs = S.sobol_designs(variables, 32, seed=3)
    assert len(designs) == 32 and len({d.design_id for d in designs}) == 32
    for index, design in enumerate(designs):
        assert design.provenance == f"scrambled-sobol:seed=3:index={index}"
        for value, variable in zip(design.values, variables, strict=True):
            assert variable.lower <= value <= variable.upper


def test_dimension_limits_fail_closed() -> None:
    with pytest.raises(ValueError):
        S.direction_numbers(0)
    with pytest.raises(ValueError):
        S.direction_numbers(S.MAX_DIMENSIONS + 1)
    with pytest.raises(ValueError):
        S.sobol_points(4, S.MAX_DIMENSIONS + 1, seed=0)
