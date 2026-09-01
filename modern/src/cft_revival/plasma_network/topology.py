"""Immutable geometry-to-chain topology with complete semantic identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from json import dumps
from math import isfinite, sqrt
from typing import Protocol, runtime_checkable

from .models import NetworkDimensions, NetworkValidationError, finite_value


class NullClassification(str, Enum):
    INTERIOR_CUSP = "interior_cusp"
    FINITE_BOUNDARY_NULL = "finite_boundary_null"


class TerminalKind(str, Enum):
    CATHODE = "cathode"
    ANODE = "anode"


def provenance_hash(payload: str) -> str:
    return sha256(payload.encode("utf-8")).hexdigest()


def _validate_hash(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NetworkValidationError(f"{name} must be a lower-case SHA-256 hex digest")
    return value


def _confidence(name: str, value: float) -> float:
    converted = finite_value(name, value)
    if not 0.0 <= converted <= 1.0:
        raise NetworkValidationError(f"{name} must be in [0, 1]")
    return converted


@dataclass(frozen=True, slots=True)
class SemanticHashes:
    """All upstream and executable identities that affect topology meaning."""

    geometry_sha256: str
    material_sha256: str
    source_sha256: str
    artifact_sha256: str
    model_sha256: str
    code_sha256: str
    schema_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, _validate_hash(name, getattr(self, name))
            )


@dataclass(frozen=True, slots=True)
class UncertainScalar:
    value: float
    standard_uncertainty: float
    unit: str
    provenance_sha256: str

    def __post_init__(self) -> None:
        value = finite_value("uncertain value", self.value)
        uncertainty = finite_value("standard_uncertainty", self.standard_uncertainty)
        if uncertainty < 0.0:
            raise NetworkValidationError("standard_uncertainty must be non-negative")
        if not isinstance(self.unit, str) or not self.unit:
            raise NetworkValidationError("uncertainty unit must be a non-empty string")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "standard_uncertainty", uncertainty)
        object.__setattr__(
            self,
            "provenance_sha256",
            _validate_hash("uncertain scalar provenance", self.provenance_sha256),
        )


@dataclass(frozen=True, slots=True)
class GeometryCell:
    cell_id: str
    axial_order: int
    axial_position_m: float
    volume_m3: float
    confidence: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.cell_id, str) or not self.cell_id:
            raise NetworkValidationError("cell_id must be non-empty")
        if not isinstance(self.axial_order, int) or isinstance(self.axial_order, bool):
            raise NetworkValidationError("axial_order must be an integer")
        position = finite_value("cell axial_position_m", self.axial_position_m)
        volume = finite_value("volume_m3", self.volume_m3)
        if volume <= 0.0:
            raise NetworkValidationError("volume_m3 must be positive")
        object.__setattr__(self, "axial_position_m", position)
        object.__setattr__(self, "volume_m3", volume)
        object.__setattr__(self, "confidence", _confidence("cell confidence", self.confidence))
        object.__setattr__(
            self,
            "provenance_sha256",
            _validate_hash("cell provenance", self.provenance_sha256),
        )


@dataclass(frozen=True, slots=True)
class GeometryNull:
    null_id: str
    axial_position_m: float
    upstream_cell_id: str | None
    downstream_cell_id: str | None
    classification: NullClassification
    loss_probability: UncertainScalar | None
    mirror_ratio: UncertainScalar | None
    exclusion_reason: str | None
    confidence: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.null_id, str) or not self.null_id:
            raise NetworkValidationError("null_id must be non-empty")
        object.__setattr__(
            self,
            "axial_position_m",
            finite_value("null axial_position_m", self.axial_position_m),
        )
        object.__setattr__(
            self, "confidence", _confidence("null confidence", self.confidence)
        )
        if not isinstance(self.classification, NullClassification):
            raise NetworkValidationError("classification must be NullClassification")
        object.__setattr__(
            self,
            "provenance_sha256",
            _validate_hash("null provenance", self.provenance_sha256),
        )
        if self.classification is NullClassification.INTERIOR_CUSP:
            if not self.upstream_cell_id or not self.downstream_cell_id:
                raise NetworkValidationError(
                    "an interior cusp must identify upstream and downstream cells"
                )
            if self.loss_probability is None:
                raise NetworkValidationError("an interior cusp requires loss_probability")
            if (
                self.loss_probability.unit != "1"
                or not 0.0 <= self.loss_probability.value < 1.0
            ):
                raise NetworkValidationError(
                    "interior cusp loss_probability must be dimensionless in [0, 1)"
                )
            if self.exclusion_reason is not None:
                raise NetworkValidationError("an interior cusp cannot have an exclusion reason")
        else:
            if self.loss_probability is not None or self.mirror_ratio is not None:
                raise NetworkValidationError(
                    "a finite-boundary null cannot carry plasma loss closures"
                )
            if not isinstance(self.exclusion_reason, str) or not self.exclusion_reason:
                raise NetworkValidationError(
                    "a finite-boundary null requires an exclusion reason"
                )


@dataclass(frozen=True, slots=True)
class TerminalBoundary:
    boundary_id: str
    kind: TerminalKind
    adjacent_cell_id: str
    axial_position_m: float
    confidence: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.boundary_id or not self.adjacent_cell_id:
            raise NetworkValidationError("terminal IDs must be non-empty")
        if not isinstance(self.kind, TerminalKind):
            raise NetworkValidationError("terminal kind must be TerminalKind")
        object.__setattr__(
            self,
            "axial_position_m",
            finite_value("terminal axial_position_m", self.axial_position_m),
        )
        object.__setattr__(
            self, "confidence", _confidence("terminal confidence", self.confidence)
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            _validate_hash("terminal provenance", self.provenance_sha256),
        )


@dataclass(frozen=True, slots=True)
class ExcludedBoundaryNull:
    null_id: str
    axial_position_m: float
    reason: str
    confidence: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.null_id:
            raise NetworkValidationError("excluded null_id must be non-empty")
        if not self.reason:
            raise NetworkValidationError("excluded boundary null reason must be non-empty")
        object.__setattr__(
            self,
            "axial_position_m",
            finite_value("excluded null axial_position_m", self.axial_position_m),
        )
        object.__setattr__(
            self,
            "confidence",
            _confidence("excluded boundary null confidence", self.confidence),
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            _validate_hash("excluded null provenance", self.provenance_sha256),
        )


@dataclass(frozen=True, slots=True)
class GeometryTopologySnapshot:
    cells: tuple[GeometryCell, ...]
    nulls: tuple[GeometryNull, ...]
    terminals: tuple[TerminalBoundary, ...]
    loss_probability_covariance: tuple[tuple[float, ...], ...]
    semantic_hashes: SemanticHashes

    def __post_init__(self) -> None:
        if not isinstance(self.cells, tuple) or not isinstance(self.nulls, tuple):
            raise NetworkValidationError("snapshot cells and nulls must be immutable tuples")
        if not isinstance(self.terminals, tuple):
            raise NetworkValidationError("snapshot terminals must be an immutable tuple")
        if not isinstance(self.semantic_hashes, SemanticHashes):
            raise NetworkValidationError("snapshot semantic_hashes must be SemanticHashes")
        if not all(isinstance(cell, GeometryCell) for cell in self.cells):
            raise NetworkValidationError("every snapshot cell must be GeometryCell")
        if not all(isinstance(null, GeometryNull) for null in self.nulls):
            raise NetworkValidationError("every snapshot null must be GeometryNull")
        if not all(
            isinstance(terminal, TerminalBoundary) for terminal in self.terminals
        ):
            raise NetworkValidationError(
                "every snapshot terminal must be TerminalBoundary"
            )


@runtime_checkable
class GeometryTopologyAdapter(Protocol):
    def plasma_topology_snapshot(self) -> GeometryTopologySnapshot:
        """Return geometry-identified cells, classified nulls, and terminals."""


@dataclass(frozen=True, slots=True)
class InteriorCusp:
    cusp_id: str
    axial_position_m: float
    upstream_cell_id: str
    downstream_cell_id: str
    loss_probability: UncertainScalar
    mirror_ratio: UncertainScalar | None
    confidence: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.cusp_id or not self.upstream_cell_id or not self.downstream_cell_id:
            raise NetworkValidationError("interior cusp IDs must be non-empty")
        object.__setattr__(
            self,
            "axial_position_m",
            finite_value("cusp axial_position_m", self.axial_position_m),
        )
        if not isinstance(self.loss_probability, UncertainScalar):
            raise NetworkValidationError("interior cusp requires UncertainScalar loss")
        self.loss_probability.__post_init__()
        if (
            self.loss_probability.unit != "1"
            or not 0.0 <= self.loss_probability.value < 1.0
        ):
            raise NetworkValidationError(
                "interior cusp loss_probability must be dimensionless in [0, 1)"
            )
        if self.mirror_ratio is not None:
            self.mirror_ratio.__post_init__()
            if self.mirror_ratio.unit != "1" or self.mirror_ratio.value <= 0.0:
                raise NetworkValidationError(
                    "mirror_ratio must be positive and dimensionless"
                )
        object.__setattr__(
            self, "confidence", _confidence("cusp confidence", self.confidence)
        )
        object.__setattr__(
            self,
            "provenance_sha256",
            _validate_hash("cusp provenance", self.provenance_sha256),
        )


def _validate_covariance(
    covariance: tuple[tuple[float, ...], ...], size: int
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(covariance, tuple) or any(
        not isinstance(row, tuple) for row in covariance
    ):
        raise NetworkValidationError("loss covariance must be an immutable tuple matrix")
    if len(covariance) != size or any(len(row) != size for row in covariance):
        raise NetworkValidationError("loss covariance must have shape (N-1, N-1)")
    matrix = tuple(
        tuple(finite_value(f"loss covariance[{i}][{j}]", value) for j, value in enumerate(row))
        for i, row in enumerate(covariance)
    )
    for i in range(size):
        if matrix[i][i] < 0.0:
            raise NetworkValidationError("loss covariance diagonal must be non-negative")
        for j in range(size):
            if matrix[i][j] != matrix[j][i]:
                raise NetworkValidationError("loss covariance must be exactly symmetric")
            if abs(matrix[i][j]) > sqrt(matrix[i][i] * matrix[j][j]):
                raise NetworkValidationError("loss covariance violates covariance bounds")
    return matrix


@dataclass(frozen=True, slots=True)
class PlasmaChainTopology:
    cells: tuple[GeometryCell, ...]
    interior_cusps: tuple[InteriorCusp, ...]
    cathode: TerminalBoundary
    anode: TerminalBoundary
    excluded_boundary_nulls: tuple[ExcludedBoundaryNull, ...]
    loss_probability_covariance: tuple[tuple[float, ...], ...]
    semantic_hashes: SemanticHashes

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.cells, tuple) or len(self.cells) < 1:
            raise NetworkValidationError("topology cells must be a non-empty immutable tuple")
        if not isinstance(self.interior_cusps, tuple):
            raise NetworkValidationError("interior_cusps must be an immutable tuple")
        if not isinstance(self.excluded_boundary_nulls, tuple):
            raise NetworkValidationError("excluded_boundary_nulls must be an immutable tuple")
        if not isinstance(self.semantic_hashes, SemanticHashes):
            raise NetworkValidationError("semantic_hashes must be SemanticHashes")
        if not all(isinstance(cell, GeometryCell) for cell in self.cells):
            raise NetworkValidationError("every topology cell must be GeometryCell")
        if not all(isinstance(cusp, InteriorCusp) for cusp in self.interior_cusps):
            raise NetworkValidationError("every topology cusp must be InteriorCusp")
        if not all(
            isinstance(item, ExcludedBoundaryNull)
            for item in self.excluded_boundary_nulls
        ):
            raise NetworkValidationError(
                "every excluded boundary null must be ExcludedBoundaryNull"
            )
        if not isinstance(self.cathode, TerminalBoundary) or not isinstance(
            self.anode, TerminalBoundary
        ):
            raise NetworkValidationError("topology terminals must be TerminalBoundary")
        self.semantic_hashes.__post_init__()
        for cell in self.cells:
            cell.__post_init__()
        for cusp in self.interior_cusps:
            cusp.__post_init__()
        for item in self.excluded_boundary_nulls:
            item.__post_init__()
        self.cathode.__post_init__()
        self.anode.__post_init__()
        dimensions = NetworkDimensions.for_cells(len(self.cells))
        if len(self.interior_cusps) != dimensions.interior_cusp_count:
            raise NetworkValidationError("topology must contain exactly N-1 interior cusps")
        cell_ids = tuple(cell.cell_id for cell in self.cells)
        if len(set(cell_ids)) != len(cell_ids):
            raise NetworkValidationError("cell IDs must be unique")
        if tuple(cell.axial_order for cell in self.cells) != tuple(range(len(self.cells))):
            raise NetworkValidationError("cells must have contiguous ordered axial_order")
        positions = tuple(cell.axial_position_m for cell in self.cells)
        if any(right <= left for left, right in zip(positions, positions[1:])):
            raise NetworkValidationError("cell axial positions must be strictly increasing")
        cusp_ids = tuple(cusp.cusp_id for cusp in self.interior_cusps)
        excluded_ids = tuple(item.null_id for item in self.excluded_boundary_nulls)
        if self.excluded_boundary_nulls != tuple(
            sorted(
                self.excluded_boundary_nulls,
                key=lambda item: (item.axial_position_m, item.null_id),
            )
        ):
            raise NetworkValidationError("excluded boundary nulls must be canonically ordered")
        all_ids = (*cell_ids, *cusp_ids, *excluded_ids, self.cathode.boundary_id, self.anode.boundary_id)
        if len(set(all_ids)) != len(all_ids):
            raise NetworkValidationError(
                "cell, cusp, excluded-null, and boundary IDs must be globally unique"
            )
        for index, cusp in enumerate(self.interior_cusps):
            if (
                cusp.upstream_cell_id != cell_ids[index]
                or cusp.downstream_cell_id != cell_ids[index + 1]
            ):
                raise NetworkValidationError(
                    "interior cusps must connect each adjacent ordered cell exactly once"
                )
            if not (
                positions[index] < cusp.axial_position_m < positions[index + 1]
            ):
                raise NetworkValidationError(
                    "each cusp position must lie strictly between its adjacent cells"
                )
        if self.cathode.kind is not TerminalKind.CATHODE:
            raise NetworkValidationError("cathode terminal has the wrong kind")
        if self.anode.kind is not TerminalKind.ANODE:
            raise NetworkValidationError("anode terminal has the wrong kind")
        if self.cathode.adjacent_cell_id != cell_ids[0]:
            raise NetworkValidationError("cathode must be adjacent to the first cell")
        if self.anode.adjacent_cell_id != cell_ids[-1]:
            raise NetworkValidationError("anode must be adjacent to the final cell")
        if not self.cathode.axial_position_m < positions[0]:
            raise NetworkValidationError("cathode must precede the first cell")
        if not self.anode.axial_position_m > positions[-1]:
            raise NetworkValidationError("anode must follow the final cell")
        _validate_covariance(
            self.loss_probability_covariance,
            dimensions.interior_cusp_count,
        )

    @property
    def dimensions(self) -> NetworkDimensions:
        self.validate()
        return NetworkDimensions.for_cells(len(self.cells))

    @property
    def excluded_finite_boundary_null_ids(self) -> tuple[str, ...]:
        return tuple(item.null_id for item in self.excluded_boundary_nulls)

    @property
    def identity_sha256(self) -> str:
        self.validate()
        payload = {
            "schema": "plasma-chain-topology-2.0.0",
            "semantic_hashes": asdict(self.semantic_hashes),
            "cells": [asdict(cell) for cell in self.cells],
            "interior_cusps": [asdict(cusp) for cusp in self.interior_cusps],
            "loss_probability_covariance": self.loss_probability_covariance,
            "cathode": asdict(self.cathode),
            "anode": asdict(self.anode),
            "excluded_boundary_nulls": [
                asdict(item) for item in self.excluded_boundary_nulls
            ],
        }
        canonical = dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return provenance_hash(canonical)


def validate_topology(topology: object) -> PlasmaChainTopology:
    if not isinstance(topology, PlasmaChainTopology):
        raise NetworkValidationError("topology must be PlasmaChainTopology")
    topology.validate()
    return topology


def build_chain_topology(adapter: GeometryTopologyAdapter) -> PlasmaChainTopology:
    if not isinstance(adapter, GeometryTopologyAdapter):
        raise NetworkValidationError("adapter must implement GeometryTopologyAdapter")
    snapshot = adapter.plasma_topology_snapshot()
    if not isinstance(snapshot, GeometryTopologySnapshot):
        raise NetworkValidationError("adapter returned an invalid topology snapshot")
    if len(snapshot.cells) < 1:
        raise NetworkValidationError("geometry must identify at least one plasma cell")
    cells = tuple(sorted(snapshot.cells, key=lambda cell: cell.axial_order))
    interior_nulls = tuple(
        item for item in snapshot.nulls
        if item.classification is NullClassification.INTERIOR_CUSP
    )
    excluded = tuple(
        ExcludedBoundaryNull(
            null_id=item.null_id,
            axial_position_m=item.axial_position_m,
            reason=item.exclusion_reason or "",
            confidence=item.confidence,
            provenance_sha256=item.provenance_sha256,
        )
        for item in snapshot.nulls
        if item.classification is NullClassification.FINITE_BOUNDARY_NULL
    )
    dimensions = NetworkDimensions.for_cells(len(cells))
    if len(interior_nulls) != dimensions.interior_cusp_count:
        raise NetworkValidationError(
            "geometry-identified interior cusps must form exactly the N-1 chain edges"
        )
    by_pair = {
        (item.upstream_cell_id, item.downstream_cell_id): item
        for item in interior_nulls
    }
    expected_pairs = tuple(
        (cells[index].cell_id, cells[index + 1].cell_id)
        for index in range(len(cells) - 1)
    )
    if len(by_pair) != len(interior_nulls) or set(by_pair) != set(expected_pairs):
        raise NetworkValidationError(
            "interior cusps must connect each adjacent ordered cell exactly once"
        )
    cusps = tuple(
        InteriorCusp(
            cusp_id=by_pair[pair].null_id,
            axial_position_m=by_pair[pair].axial_position_m,
            upstream_cell_id=pair[0],
            downstream_cell_id=pair[1],
            loss_probability=by_pair[pair].loss_probability,  # type: ignore[arg-type]
            mirror_ratio=by_pair[pair].mirror_ratio,
            confidence=by_pair[pair].confidence,
            provenance_sha256=by_pair[pair].provenance_sha256,
        )
        for pair in expected_pairs
    )
    if len(snapshot.terminals) != 2:
        raise NetworkValidationError("exactly cathode and anode terminals are required")
    terminal_by_kind = {terminal.kind: terminal for terminal in snapshot.terminals}
    if set(terminal_by_kind) != {TerminalKind.CATHODE, TerminalKind.ANODE}:
        raise NetworkValidationError("terminals must contain one cathode and one anode")
    return PlasmaChainTopology(
        cells=cells,
        interior_cusps=cusps,
        cathode=terminal_by_kind[TerminalKind.CATHODE],
        anode=terminal_by_kind[TerminalKind.ANODE],
        excluded_boundary_nulls=tuple(
            sorted(excluded, key=lambda item: (item.axial_position_m, item.null_id))
        ),
        loss_probability_covariance=_validate_covariance(
            snapshot.loss_probability_covariance,
            dimensions.interior_cusp_count,
        ),
        semantic_hashes=snapshot.semantic_hashes,
    )


@dataclass(frozen=True, slots=True)
class SnapshotAdapter:
    snapshot: GeometryTopologySnapshot

    def plasma_topology_snapshot(self) -> GeometryTopologySnapshot:
        return self.snapshot


def make_chain_topology(
    cell_count: int,
    interior_loss_probabilities: tuple[float, ...],
    *,
    provenance_seed: str,
    loss_standard_uncertainties: tuple[float, ...] | None = None,
    loss_probability_covariance: tuple[tuple[float, ...], ...] | None = None,
    mirror_ratios: tuple[float | None, ...] | None = None,
    mirror_standard_uncertainties: tuple[float, ...] | None = None,
    finite_boundary_null_count: int = 0,
) -> PlasmaChainTopology:
    """Create explicit deterministic fixtures; production uses the adapter protocol."""

    dimensions = NetworkDimensions.for_cells(cell_count)
    if len(interior_loss_probabilities) != dimensions.interior_cusp_count:
        raise NetworkValidationError("interior_loss_probabilities must contain N-1 values")
    uncertainties = (
        (0.0,) * dimensions.interior_cusp_count
        if loss_standard_uncertainties is None
        else loss_standard_uncertainties
    )
    mirrors = (
        (None,) * dimensions.interior_cusp_count
        if mirror_ratios is None
        else mirror_ratios
    )
    mirror_uncertainties = (
        (0.0,) * dimensions.interior_cusp_count
        if mirror_standard_uncertainties is None
        else mirror_standard_uncertainties
    )
    if not (
        len(uncertainties)
        == len(mirrors)
        == len(mirror_uncertainties)
        == dimensions.interior_cusp_count
    ):
        raise NetworkValidationError("uncertainty and mirror sequences must contain N-1 values")
    covariance = (
        tuple(
            tuple(
                uncertainties[i] ** 2 if i == j else 0.0
                for j in range(dimensions.interior_cusp_count)
            )
            for i in range(dimensions.interior_cusp_count)
        )
        if loss_probability_covariance is None
        else loss_probability_covariance
    )
    cells = tuple(
        GeometryCell(
            cell_id=f"cell-{index}",
            axial_order=index,
            axial_position_m=float(index),
            volume_m3=1.0,
            confidence=1.0,
            provenance_sha256=provenance_hash(f"{provenance_seed}:cell:{index}"),
        )
        for index in range(cell_count)
    )
    nulls: list[GeometryNull] = []
    for index, probability in enumerate(interior_loss_probabilities):
        loss = UncertainScalar(
            probability,
            uncertainties[index],
            "1",
            provenance_hash(f"{provenance_seed}:loss:{index}"),
        )
        mirror_value = mirrors[index]
        mirror = (
            None
            if mirror_value is None
            else UncertainScalar(
                mirror_value,
                mirror_uncertainties[index],
                "1",
                provenance_hash(f"{provenance_seed}:mirror:{index}"),
            )
        )
        nulls.append(
            GeometryNull(
                null_id=f"cusp-{index}",
                axial_position_m=index + 0.5,
                upstream_cell_id=cells[index].cell_id,
                downstream_cell_id=cells[index + 1].cell_id,
                classification=NullClassification.INTERIOR_CUSP,
                loss_probability=loss,
                mirror_ratio=mirror,
                exclusion_reason=None,
                confidence=1.0,
                provenance_sha256=provenance_hash(f"{provenance_seed}:cusp:{index}"),
            )
        )
    for index in range(finite_boundary_null_count):
        nulls.append(
            GeometryNull(
                null_id=f"finite-boundary-null-{index}",
                axial_position_m=cell_count + float(index),
                upstream_cell_id=None,
                downstream_cell_id=None,
                classification=NullClassification.FINITE_BOUNDARY_NULL,
                loss_probability=None,
                mirror_ratio=None,
                exclusion_reason="finite computational boundary classification",
                confidence=1.0,
                provenance_sha256=provenance_hash(
                    f"{provenance_seed}:boundary-null:{index}"
                ),
            )
        )
    terminals = (
        TerminalBoundary(
            "cathode",
            TerminalKind.CATHODE,
            cells[0].cell_id,
            -0.5,
            1.0,
            provenance_hash(f"{provenance_seed}:cathode"),
        ),
        TerminalBoundary(
            "anode",
            TerminalKind.ANODE,
            cells[-1].cell_id,
            cell_count - 0.5,
            1.0,
            provenance_hash(f"{provenance_seed}:anode"),
        ),
    )
    semantic_hashes = SemanticHashes(
        **{
            f"{name}_sha256": provenance_hash(f"{provenance_seed}:{name}")
            for name in (
                "geometry",
                "material",
                "source",
                "artifact",
                "model",
                "code",
                "schema",
            )
        }
    )
    return build_chain_topology(
        SnapshotAdapter(
            GeometryTopologySnapshot(
                cells,
                tuple(nulls),
                terminals,
                covariance,
                semantic_hashes,
            )
        )
    )
