import pytest

from cft_revival.plasma_v2 import (
    AnodeRow,
    CuspLossClosure,
    CuspSheathSpec,
    FourthPotentialRow,
    PotentialClosure,
    SheathClosureInputs,
    SheathRegime,
)
from cft_revival.plasma_v2.targets import KORNFELD_DM92

KORNFELD_P = (0.060, 0.119, 0.160, 0.254)


@pytest.fixture
def scl_cusps() -> tuple[CuspSheathSpec, CuspSheathSpec, CuspSheathSpec]:
    return tuple(CuspSheathSpec(regime=SheathRegime.SPACE_CHARGE_LIMITED) for _ in range(3))  # type: ignore[return-value]


@pytest.fixture
def no_emission_cusps() -> tuple[CuspSheathSpec, CuspSheathSpec, CuspSheathSpec]:
    return tuple(CuspSheathSpec() for _ in range(3))  # type: ignore[return-value]


@pytest.fixture
def kornfeld_mode_a(scl_cusps) -> SheathClosureInputs:
    """1 kV / 1 A, Kornfeld p, flat interior, anode fall 0 V declared, phi_1 solved (R31 sheath)."""

    return SheathClosureInputs(
        1000.0,
        1.0,
        scl_cusps,
        KORNFELD_P[3],
        cusp_loss_closure=CuspLossClosure.CL1_DECLARED,
        declared_cusp_probabilities=KORNFELD_P[:3],
        potentials=PotentialClosure(),
    )


@pytest.fixture
def kornfeld_mode_c(scl_cusps) -> SheathClosureInputs:
    """1 kV / 1 A, Kornfeld p and Kornfeld's own potentials declared."""

    return SheathClosureInputs(
        1000.0,
        1.0,
        scl_cusps,
        KORNFELD_P[3],
        cusp_loss_closure=CuspLossClosure.CL1_DECLARED,
        declared_cusp_probabilities=KORNFELD_P[:3],
        potentials=PotentialClosure(
            anode_row=AnodeRow.DECLARED_FALL,
            fourth_row=FourthPotentialRow.CATHODE_COUPLING_DECLARED,
            cathode_coupling_v=14.1,
        ),
    )


@pytest.fixture
def kornfeld_target():
    return KORNFELD_DM92
