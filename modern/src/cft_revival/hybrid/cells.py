"""Control volumes of the L2 v2 per-cell hybrid: the cusp-cell catalogue cells of a design.

The cusp topology search v3.1 (``experiments/cusp_topology_search_v3_1``) defines, for every
design, the wall cusps (separatrix of an axis null -> intersection with the straight
dielectric wall) and the cells they bound.  L2 v2 uses those cells as its electron control
volumes: one electron fluid state (count, energy) per cell, one Boltzmann reference per
cell, cusp conductances between neighbouring cells.  The catalogue covers the straight
dielectric only (``straight_z_min_m``..``straight_z_max_m``); the model extends the
anode-side partial cell to the anode plane and the exit-side partial cell to the exit plane
(injector zone and divergent cone belong to the end cells), exactly as the design
mini-sweep's Kornfeld mapping does ("model cell 1 = exit-side partial cell + cone + plume").

``load_reference_partition`` verifies the catalogue bytes against the sealed bundle manifest
(the same check as the catalogue's own loader) and, when the caller declares the PIC's cusp
planes, refuses a catalogue whose planes differ from them by more than the declared
tolerance: the L2 cells and the PIC's cusp overlays must be the same planes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..experiment_runtime import strict_json_file
from ..pic2d.models import Grid2D
from .models import HybridValidationError

CATALOGUE_SCHEMA = "cft-revival.cusp-cell-catalogue/1.0.0"
CATALOGUE_RELATIVE_PATH = "artifacts/cusp-cell-catalogue.json"
CATALOGUE_LABELS = (
    "SCREENING_L1A_FIELD_SEPARATRIX_CUSP_TOPOLOGY",
    "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY",
)


@dataclass(frozen=True, slots=True)
class CellPartition:
    """Axial partition of the plasma domain into electron control volumes.

    ``z_start_m``/``z_end_m`` tile ``[z_min, z_max]`` of the simulated domain; cell ``k``
    is ``[z_start_m[k], z_end_m[k])`` (the last cell is closed at ``z_max``).  Cells are
    ordered anode -> exit, so ``cusp_z_m[k]`` is the plane between cells ``k`` and ``k+1``.
    """

    design_id: str
    set_id: str
    label: str
    cell_ids: tuple[str, ...]
    kinds: tuple[str, ...]
    z_start_m: tuple[float, ...]
    z_end_m: tuple[float, ...]
    cusp_z_m: tuple[float, ...]
    catalogue_sha256: str | None
    source: str

    def __post_init__(self) -> None:
        n = len(self.cell_ids)
        if n < 1 or len(self.kinds) != n or len(self.z_start_m) != n or len(self.z_end_m) != n:
            raise HybridValidationError("cell partition arrays must share one length >= 1")
        if len(self.cusp_z_m) != n - 1:
            raise HybridValidationError("a partition of n cells has n - 1 cusp planes")
        for k in range(n):
            if not self.z_end_m[k] > self.z_start_m[k]:
                raise HybridValidationError(f"cell {self.cell_ids[k]} has non-positive length")
            if k > 0 and self.z_start_m[k] != self.z_end_m[k - 1]:
                raise HybridValidationError("cells must tile the axis without gaps or overlaps")
            if k < n - 1 and abs(self.cusp_z_m[k] - self.z_end_m[k]) > 1e-12:
                raise HybridValidationError("cusp planes must coincide with the cell boundaries")
        if self.label not in CATALOGUE_LABELS and self.label != "SYNTHETIC_PARTITION":
            raise HybridValidationError(f"unknown partition label {self.label!r}")

    @property
    def cell_count(self) -> int:
        return len(self.cell_ids)

    @property
    def z_min_m(self) -> float:
        return self.z_start_m[0]

    @property
    def z_max_m(self) -> float:
        return self.z_end_m[-1]

    def cell_of_z(self, z_m: np.ndarray) -> np.ndarray:
        """Cell index per axial coordinate (half-open intervals; the last cell is closed).

        Coordinates outside ``[z_min, z_max]`` (by more than 1e-12) map to -1.
        """

        z = np.asarray(z_m, dtype=np.float64)
        boundaries = np.array([*self.z_start_m, self.z_end_m[-1]], dtype=np.float64)
        index = np.searchsorted(boundaries, z + 1e-12, side="right") - 1
        index = np.clip(index, -1, self.cell_count - 1)
        index = np.where(z > boundaries[-1] + 1e-12, -1, index)
        index = np.where(z < boundaries[0] - 1e-12, -1, index)
        return index.astype(np.int64)

    def node_cells(self, grid: Grid2D) -> np.ndarray:
        """Cell index of every node column, broadcast to the ``(nr+1, nz+1)`` node array."""

        column = self.cell_of_z(grid.z_m)
        if np.any(column < 0):
            raise HybridValidationError("the partition does not cover the grid's axial extent")
        return np.broadcast_to(column[None, :], grid.node_shape).copy()

    def cusp_columns(self, grid: Grid2D) -> tuple[int, ...]:
        """Nearest node column of each cusp plane (diagnostic overlay only)."""

        return tuple(round((z - grid.geometry.z_min_m) / grid.dz_m) for z in self.cusp_z_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "set_id": self.set_id,
            "label": self.label,
            "source": self.source,
            "catalogue_sha256": self.catalogue_sha256,
            "cells": [
                {"cell_id": c, "kind": k, "z_start_m": a, "z_end_m": b}
                for c, k, a, b in zip(self.cell_ids, self.kinds, self.z_start_m, self.z_end_m, strict=True)
            ],
            "cusp_z_m": list(self.cusp_z_m),
        }


def synthetic_partition(z_min_m: float, z_max_m: float, cusp_z_m: Sequence[float]) -> CellPartition:
    """A declared partition for tests and shakedowns (never a physical claim)."""

    cusps = tuple(sorted(float(z) for z in cusp_z_m))
    if any(not z_min_m < z < z_max_m for z in cusps):
        raise HybridValidationError("synthetic cusp planes must lie strictly inside the domain")
    starts = (float(z_min_m), *cusps)
    ends = (*cusps, float(z_max_m))
    n = len(starts)
    kinds = tuple("anode_partial" if k == 0 else "exit_partial" if k == n - 1 else "interior" for k in range(n))
    if n == 1:
        kinds = ("unbounded",)
    return CellPartition(
        design_id="synthetic", set_id="synthetic", label="SYNTHETIC_PARTITION",
        cell_ids=tuple(f"cell-{k + 1:02d}" for k in range(n)), kinds=kinds,
        z_start_m=starts, z_end_m=ends, cusp_z_m=cusps, catalogue_sha256=None, source="synthetic_partition",
    )


def load_sealed_catalogue(results_root: Path) -> tuple[dict[str, Any], str]:
    """The catalogue only if its bytes are the ones sealed in the bundle manifest (returns it with its SHA-256)."""

    manifest = strict_json_file(results_root / "manifest.json")
    if manifest.get("state") != "accepted_result":
        raise HybridValidationError("the catalogue's bundle is not an accepted result")
    entry = next((item for item in manifest["artifacts"] if item.get("path") == CATALOGUE_RELATIVE_PATH), None)
    if entry is None:
        raise HybridValidationError("catalogue is not listed in the bundle manifest")
    raw = (results_root / CATALOGUE_RELATIVE_PATH).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["byte_sha256"] or len(raw) != entry["bytes"]:
        raise HybridValidationError("catalogue bytes differ from the sealed manifest entry")
    catalogue = strict_json_file(results_root / CATALOGUE_RELATIVE_PATH)
    if catalogue.get("schema_version") != CATALOGUE_SCHEMA:
        raise HybridValidationError("catalogue schema version differs from the contract")
    return catalogue, digest


def partition_from_entry(
    entry: Mapping[str, Any],
    *,
    z_min_m: float,
    z_max_m: float,
    catalogue_sha256: str | None,
    source: str,
) -> CellPartition:
    """Catalogue cells of one design extended to the simulated domain ``[z_min, z_max]``."""

    cells = sorted(entry["cells"], key=lambda c: float(c["z_start_m"]))
    if not cells:
        raise HybridValidationError("catalogue entry has no cells")
    geometry = entry["geometry"]
    if not (z_min_m <= float(geometry["straight_z_min_m"]) and float(geometry["straight_z_max_m"]) <= z_max_m):
        raise HybridValidationError("the simulated domain must contain the catalogue's straight dielectric")
    starts = [float(c["z_start_m"]) for c in cells]
    ends = [float(c["z_end_m"]) for c in cells]
    starts[0] = float(z_min_m)
    ends[-1] = float(z_max_m)
    cusps = tuple(float(c["z_c_m"]) for c in entry["wall_cusps"])
    if len(cusps) != len(cells) - 1:
        raise HybridValidationError("catalogue entry cusps and cells are inconsistent")
    return CellPartition(
        design_id=str(entry["design_id"]), set_id=str(entry["set_id"]), label=str(entry["label"]),
        cell_ids=tuple(str(c["cell_id"]) for c in cells), kinds=tuple(str(c["kind"]) for c in cells),
        z_start_m=tuple(starts), z_end_m=tuple(ends), cusp_z_m=cusps,
        catalogue_sha256=catalogue_sha256, source=source,
    )


def load_reference_partition(
    results_root: Path,
    *,
    set_id: str,
    design_id: str,
    grid: Grid2D,
    declared_cusp_planes_m: Sequence[float] | None = None,
    plane_tolerance_m: float = 2.5e-5,
) -> CellPartition:
    """Cells of ``set_id:design_id`` from the sealed v3.1 catalogue on the PIC domain of ``grid``.

    With ``declared_cusp_planes_m`` (the PIC's cusp planes, e.g. 6.028 / 12.000 / 17.972 mm for
    the reference design) the catalogue planes must agree within ``plane_tolerance_m``
    (default: half a 50 um PIC cell); otherwise the cells are not the PIC's cusp cells and the
    loader refuses.
    """

    catalogue, digest = load_sealed_catalogue(results_root)
    entry = next((e for e in catalogue["entries"] if e["set_id"] == set_id and e["design_id"] == design_id), None)
    if entry is None:
        raise HybridValidationError(f"{set_id}:{design_id} is not in the catalogue")
    if not entry["stable"]:
        raise HybridValidationError(f"{set_id}:{design_id} is not a stable catalogue entry")
    geometry = grid.geometry
    partition = partition_from_entry(
        entry, z_min_m=geometry.z_min_m, z_max_m=geometry.domain_z_max_m, catalogue_sha256=digest,
        source=f"{results_root.as_posix()}/{CATALOGUE_RELATIVE_PATH}",
    )
    if declared_cusp_planes_m is not None:
        declared = tuple(sorted(float(z) for z in declared_cusp_planes_m))
        if len(declared) != len(partition.cusp_z_m):
            raise HybridValidationError(
                f"the catalogue has {len(partition.cusp_z_m)} wall cusps, the declaration {len(declared)}"
            )
        differences = [abs(a - b) for a, b in zip(declared, partition.cusp_z_m, strict=True)]
        if max(differences) > plane_tolerance_m:
            raise HybridValidationError(
                f"catalogue cusp planes {partition.cusp_z_m} differ from the declared PIC planes {declared} by up to "
                f"{max(differences):.3e} m (> {plane_tolerance_m:.1e} m)"
            )
    return partition


__all__ = [
    "CATALOGUE_LABELS",
    "CATALOGUE_RELATIVE_PATH",
    "CATALOGUE_SCHEMA",
    "CellPartition",
    "load_reference_partition",
    "load_sealed_catalogue",
    "partition_from_entry",
    "synthetic_partition",
]
