"""Backend contracts and legacy FEMM export compatibility."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import (
    CuspProbabilities,
    DesignPoint,
    FieldProfile,
    MagneticFieldResult,
    PlasmaSolution,
    ValidationError,
)


class UnknownPhysicsError(RuntimeError):
    """Raised where legacy intent is insufficient for a safe translation."""


class MagneticFieldBackend(ABC):
    @abstractmethod
    def solve(self, design: DesignPoint, generation: int, individual: int) -> MagneticFieldResult:
        """Produce field profiles for one design point."""


class PlasmaBackend(ABC):
    @abstractmethod
    def solve(
        self, design: DesignPoint, probabilities: CuspProbabilities
    ) -> PlasmaSolution:
        """Produce a validated 30-variable plasma solution."""


def _read_femm_plot(path: Path) -> FieldProfile:
    if not path.is_file():
        raise FileNotFoundError(path)
    positions: list[float] = []
    magnitudes: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.strip().split()
        if len(columns) < 2:
            continue
        try:
            position, magnitude = float(columns[0]), float(columns[1])
        except ValueError:
            # Legacy FEMM plots contain two header rows.
            continue
        positions.append(position)
        magnitudes.append(abs(magnitude))
    if not positions:
        raise ValidationError(f"no numeric FEMM samples found in {path}")
    return FieldProfile(tuple(positions), tuple(magnitudes))


class FemmExportBackend(MagneticFieldBackend):
    """Read files produced by the legacy `mo_makeplot` calls.

    This adapter deliberately does not automate FEMM. A future Windows worker
    may wrap FEMM, but it must serialize access and atomically publish these
    files before this process reads them.
    """

    def __init__(self, export_directory: Path) -> None:
        self._export_directory = export_directory

    def solve(self, design: DesignPoint, generation: int, individual: int) -> MagneticFieldResult:
        design.validate()
        suffix = f"_Gen_{generation}_id_{individual}.txt"
        centreline_path = (
            self._export_directory / f"Flux Magnitude Channel Centreline{suffix}"
        )
        wall_path = self._export_directory / f"Flux Magnitude Channel Wall{suffix}"
        return MagneticFieldResult(
            centreline=_read_femm_plot(centreline_path),
            wall=_read_femm_plot(wall_path),
            provenance=f"legacy-femm-export:{centreline_path.name},{wall_path.name}",
        )


class UnimplementedPlasmaBackend(PlasmaBackend):
    """Safety stop until the equation set is independently re-derived."""

    def solve(
        self, design: DesignPoint, probabilities: CuspProbabilities
    ) -> PlasmaSolution:
        del design, probabilities
        raise UnknownPhysicsError(
            "plasma equations are quarantined: verify signs, globals, constraints, "
            "and normalization against Kornfeld et al. before translation"
        )
