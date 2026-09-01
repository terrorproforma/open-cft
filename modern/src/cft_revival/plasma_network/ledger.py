"""Programmatic one-row/one-ID equation and source ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .models import NetworkDimensions
from .topology import PlasmaChainTopology, validate_topology

EquationUnit = Literal["A", "W"]


@dataclass(frozen=True, slots=True)
class SourceLedgerEntry:
    source_id: str
    role: str
    physical_claim_status: str


@dataclass(frozen=True, slots=True)
class EquationDefinition:
    row_index: int
    row_id: str
    equation_id: str
    family: str
    unit: EquationUnit
    normalization: str
    orientation: str
    expression: str
    source_ids: tuple[str, ...]


SOURCE_LEDGER = (
    SourceLedgerEntry(
        source_id="KORNFELD-IEPC-2007-108",
        role="lineage for conditional current/power balance form",
        physical_claim_status="equation lineage only; not experimental validation",
    ),
    SourceLedgerEntry(
        source_id="ACCEPTED-CORRECTED-FOUR-CELL",
        role="N=4 executable compatibility oracle",
        physical_claim_status="software parity only; not a physical claim",
    ),
    SourceLedgerEntry(
        source_id="GEOMETRY-TOPOLOGY-ADAPTER",
        role="cell/cusp incidence, uncertainty, and provenance",
        physical_claim_status="adapter evidence must be supplied by caller",
    ),
)


def generate_equation_ledger(
    topology: PlasmaChainTopology,
) -> tuple[EquationDefinition, ...]:
    """Generate deterministic rows; N=4 indices match the accepted residual."""

    validate_topology(topology)
    n = topology.dimensions.cell_count
    rows: list[EquationDefinition] = []

    def add(
        equation_id: str,
        family: str,
        unit: EquationUnit,
        orientation: str,
        expression: str,
        sources: tuple[str, ...] = (
            "KORNFELD-IEPC-2007-108",
            "ACCEPTED-CORRECTED-FOUR-CELL",
        ),
    ) -> None:
        index = len(rows)
        rows.append(
            EquationDefinition(
                row_index=index,
                row_id=f"R{index:02d}",
                equation_id=equation_id,
                family=family,
                unit=unit,
                normalization="anode_current_a" if unit == "A" else "anode_input_power_w",
                orientation=orientation,
                expression=expression,
                source_ids=sources,
            )
        )

    add(
        "cathode.electron_emission",
        "cathode_boundary",
        "A",
        "electron current is positive from cathode toward anode",
        "je[0] - P*(phi[0]-phi_cathode)^(3/2) = 0",
    )
    for cell in range(n - 1):
        add(
            f"cell.{cell}.electron_continuity",
            "electron_continuity",
            "A",
            "positive downstream; interior cusp loss leaves the axial control volume",
            f"je[{cell + 1}] - je[{cell}]*(1-p[{cell}]) - I[{cell}] = 0",
        )
    for cell in range(n):
        add(
            f"cell.{cell}.ionization",
            "ionization_source",
            "A",
            "positive source creates equal electron and ion current",
            f"I[{cell}] - je[{cell}]*(1-p[{cell}])*CI*dE[{cell}]/EI = 0",
        )
    for cell in range(n - 1):
        add(
            f"cell.{cell}.ion_continuity",
            "ion_continuity",
            "A",
            "ion current is positive toward cathode; cusp current is outward",
            f"ji[{cell}] - ji[{cell + 1}] - I[{cell}] + jic[{cell}] = 0",
        )
    add(
        "anode.ion_boundary",
        "anode_boundary",
        "A",
        "terminal ion current retains its sign",
        f"ji[{n - 1}] - I[{n - 1}] - ji[{n}] = 0",
    )
    for cell in range(1, n):
        add(
            f"cell.{cell}.thermal_transport",
            "thermal_transport",
            "W",
            "electron energy transport is positive downstream",
            f"Te[{cell}]*(je[{cell}]*(1-p[{cell}])+I[{cell}]) "
            f"- CT*je[{cell}]*(1-p[{cell}])*dE[{cell}] = 0",
        )
    for interface in range(n + 1):
        add(
            f"interface.{interface}.current",
            "interface_current",
            "A",
            "electron current downstream plus ion current upstream equals imposed current",
            f"je[{interface}] + ji[{interface}] - Ia = 0",
        )
    for cusp in range(n - 1):
        add(
            f"cusp.{cusp}.ion_loss",
            "cusp_loss",
            "A",
            "positive current leaves the axial ion control volume",
            f"jic[{cusp}] - p[{cusp}]*je[{cusp}] = 0",
            (
                "KORNFELD-IEPC-2007-108",
                "ACCEPTED-CORRECTED-FOUR-CELL",
                "GEOMETRY-TOPOLOGY-ADAPTER",
            ),
        )
    for cell in range(n):
        add(
            f"cell.{cell}.energy",
            "cell_energy",
            "W",
            "received electron power equals thermal, ionization, and excitation losses",
            f"(1-CE)*je[{cell}]*(1-p[{cell}])*dE[{cell}] "
            f"- (je[{cell}]*(1-p[{cell}])+I[{cell}])*Te[{cell}] - I[{cell}]*EI = 0",
        )
    add(
        "network.global_energy",
        "global_energy",
        "W",
        "all named losses and signed terminal exchange minus anode input",
        "Pbeam + PI + PE + Pcusp + Panode - Ua*Ia = 0",
    )
    dimensions = NetworkDimensions.for_cells(n)
    if len(rows) != dimensions.residual_size:
        raise AssertionError("equation generator dimension invariant failed")
    if len({row.row_id for row in rows}) != len(rows):
        raise AssertionError("equation row IDs are not one-to-one")
    if len({row.equation_id for row in rows}) != len(rows):
        raise AssertionError("semantic equation IDs are not one-to-one")
    return tuple(rows)


def machine_ledger(topology: PlasmaChainTopology) -> dict[str, object]:
    """Return a JSON-serializable generated ledger with topology identity."""

    return {
        "schema_version": "1.0.0",
        "topology_identity_sha256": topology.identity_sha256,
        "dimensions": asdict(topology.dimensions),
        "sources": [asdict(item) for item in SOURCE_LEDGER],
        "equations": [asdict(item) for item in generate_equation_ledger(topology)],
        "inequalities_are_residual_rows": False,
    }
