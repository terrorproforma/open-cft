"""Shared fixtures for the axisymmetric PIC-MCC tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d.models import ChannelGeometry, Grid2D

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# The divergent-exit channel used by the orbit campaign (read from the P2 mesh):
# bore r = 2 mm, straight to z = 18 mm, linear cone to r = 3 mm at z = 24 mm.
CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
STRAIGHT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 8.0e-3, 2.0e-3)


@pytest.fixture(scope="session")
def cft_geometry() -> ChannelGeometry:
    return CFT_GEOMETRY


@pytest.fixture(scope="session")
def coarse_cft_grid() -> Grid2D:
    return Grid2D(CFT_GEOMETRY, 12, 96)


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def p2_available(repository_root: Path) -> bool:
    checkpoint = (
        repository_root / "modern" / "examples" / "fem_reference" / "artifacts" / "third-level"
        / "divergent-exit-stack" / "checkpoints" / "divergent-exit-stack.level-1.json"
    )
    return checkpoint.is_file() and checkpoint.stat().st_size > 1_000_000


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260903)
