"""Explicit boundaries for future WarpX/PICMI and LXCat integrations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from .collisions import CrossSectionTable
from .models import Grid1D, PICConfig, ParticleState, Species


@dataclass(frozen=True, slots=True)
class PICMIProblem:
    """Backend-neutral input packet; no WarpX object leaks into the core."""

    grid: Grid1D
    species: Species
    particles: ParticleState
    config: PICConfig


class WarpXPICMIAdapter(Protocol):
    """Contract a future optional WarpX package must implement and verify."""

    backend_name: str

    def build_inputs(self, problem: PICMIProblem) -> dict[str, Any]: ...

    def run_one_step(self, problem: PICMIProblem) -> dict[str, Any]: ...


def lxcat_source_hash(path: Path) -> str:
    """Hash exact downloaded LXCat bytes before parsing or normalization."""

    return sha256(Path(path).read_bytes()).hexdigest()


class LXCatParser(Protocol):
    """Parser boundary requiring the exact source-byte digest in its result."""

    def parse(
        self,
        raw_bytes: bytes,
        *,
        source_url: str,
        source_sha256: str,
        process: str,
    ) -> CrossSectionTable: ...
