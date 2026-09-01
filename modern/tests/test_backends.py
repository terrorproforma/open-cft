from pathlib import Path

import pytest

from cft_revival.backends import (
    FemmExportBackend,
    UnimplementedPlasmaBackend,
    UnknownPhysicsError,
)
from cft_revival.models import CuspProbabilities, DesignPoint


def _write_plot(path: Path) -> None:
    path.write_text(
        "FEMM plot header\nLength\tB\n0.0\t-0.1\n0.5\t0.2\n",
        encoding="utf-8",
    )


def test_femm_export_backend_reads_legacy_names(tmp_path: Path) -> None:
    suffix = "_Gen_2_id_7.txt"
    _write_plot(tmp_path / f"Flux Magnitude Channel Centreline{suffix}")
    _write_plot(tmp_path / f"Flux Magnitude Channel Wall{suffix}")
    design = DesignPoint.from_sequence((300.0, 1.0, 10.0, 3.0, 8.0, 12.0, 20.0, 30.0))

    result = FemmExportBackend(tmp_path).solve(design, 2, 7)

    assert result.centreline.positions_mm == (0.0, 0.5)
    assert result.centreline.magnitudes_t == (0.1, 0.2)
    assert result.provenance.startswith("legacy-femm-export:")


def test_plasma_backend_fails_closed() -> None:
    design = DesignPoint.from_sequence((300.0, 1.0, 10.0, 3.0, 8.0, 12.0, 20.0, 30.0))
    with pytest.raises(UnknownPhysicsError, match="quarantined"):
        UnimplementedPlasmaBackend().solve(
            design, CuspProbabilities(0.01, 0.02, 0.03, 0.04)
        )
