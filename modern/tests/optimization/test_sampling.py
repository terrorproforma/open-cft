from dataclasses import dataclass

from cft_revival.optimization.domain import Variable
from cft_revival.optimization.sampling import (
    grouped_train_validation_split,
    initial_designs,
    shifted_halton,
)


VARIABLES = tuple(Variable(f"x{index}", -1.0, 1.0, "1") for index in range(8))


def test_shifted_halton_is_deterministic_bounded_and_not_mislabelled_sobol() -> None:
    first = shifted_halton(20, 8, seed=17)
    assert first == shifted_halton(20, 8, seed=17)
    assert first != shifted_halton(20, 8, seed=18)
    assert all(0.0 <= coordinate < 1.0 for point in first for coordinate in point)
    assert len(set(first)) == len(first)


def test_initial_designs_include_boundaries_and_exact_count() -> None:
    designs = initial_designs(VARIABLES, 24, seed=4)
    assert len(designs) == 24
    assert len({design.design_id for design in designs}) == 24
    assert tuple(designs[0].values) == (-1.0,) * 8
    assert tuple(designs[1].values) == (1.0,) * 8
    assert tuple(designs[2].values) == (0.0,) * 8
    assert all(
        design.provenance.startswith("boundary-challenge:")
        for design in designs[:21]
    )
    assert all(
        design.provenance.startswith("shifted-halton:")
        for design in designs[21:]
    )


def test_initial_designs_accepts_empty_request() -> None:
    assert initial_designs(VARIABLES, 0, seed=4) == ()


@dataclass(frozen=True)
class Row:
    design_id: str
    fidelity: str
    seed: int


def test_grouped_split_keeps_all_fidelities_and_seeds_together() -> None:
    rows = tuple(
        Row(f"design-{design}", fidelity, seed)
        for design in range(10)
        for fidelity in ("F0", "F3")
        for seed in (0, 1)
    )
    train, validation = grouped_train_validation_split(
        rows, lambda row: row.design_id, validation_fraction=0.3, seed=9
    )
    train_ids = {row.design_id for row in train}
    validation_ids = {row.design_id for row in validation}
    assert train_ids.isdisjoint(validation_ids)
    assert len(validation_ids) == 3
    assert len(train) + len(validation) == len(rows)
