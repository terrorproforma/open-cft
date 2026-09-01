from dataclasses import replace

import pytest

from cft_revival.plasma_network import (
    GeometryNull,
    GeometryTopologySnapshot,
    NetworkDimensions,
    NetworkValidationError,
    NullClassification,
    SnapshotAdapter,
    build_chain_topology,
    generate_equation_ledger,
    machine_ledger,
    make_chain_topology,
    manufactured_zero_cusp_case,
    provenance_hash,
    solve_network,
    validate_topology,
)


@pytest.mark.parametrize("cell_count", range(1, 7))
def test_dimensions_and_equation_ids_are_derived_from_topology(cell_count: int) -> None:
    topology = make_chain_topology(
        cell_count,
        (0.02,) * (cell_count - 1),
        provenance_seed=f"dimensions-{cell_count}",
    )
    dimensions = topology.dimensions
    assert dimensions == NetworkDimensions(
        cell_count=cell_count,
        interior_cusp_count=cell_count - 1,
        terminal_boundary_count=2,
        state_size=6 * cell_count + 1,
        residual_size=7 * cell_count,
            structural_rank=5 * cell_count + 2,
        structural_nullity=cell_count - 1,
    )
    rows = generate_equation_ledger(topology)
    assert len(rows) == 7 * cell_count
    assert [row.row_id for row in rows] == [
        f"R{index:02d}" for index in range(7 * cell_count)
    ]
    assert len({row.equation_id for row in rows}) == len(rows)
    assert all(row.unit in {"A", "W"} for row in rows)
    assert not machine_ledger(topology)["inequalities_are_residual_rows"]


def test_n4_ledger_preserves_accepted_row_family_order() -> None:
    topology = make_chain_topology(4, (0.1, 0.2, 0.3), provenance_seed="n4-ledger")
    families = tuple(row.family for row in generate_equation_ledger(topology))
    assert families == (
        "cathode_boundary",
        *("electron_continuity",) * 3,
        *("ionization_source",) * 4,
        *("ion_continuity",) * 3,
        "anode_boundary",
        *("thermal_transport",) * 3,
        *("interface_current",) * 5,
        *("cusp_loss",) * 3,
        *("cell_energy",) * 4,
        "global_energy",
    )


def test_geometry_adapter_excludes_finite_boundary_nulls_without_windows() -> None:
    topology = make_chain_topology(
        3,
        (0.1, 0.2),
        provenance_seed="boundary-filter",
        finite_boundary_null_count=2,
    )
    assert len(topology.interior_cusps) == 2
    assert topology.excluded_finite_boundary_null_ids == (
        "finite-boundary-null-0",
        "finite-boundary-null-1",
    )
    assert topology.interior_cusps[0].loss_probability.provenance_sha256
    assert topology.identity_sha256 == topology.identity_sha256


def test_malformed_graphs_fail_before_residual_construction() -> None:
    valid = make_chain_topology(3, (0.1, 0.2), provenance_seed="malformed")
    snapshot = GeometryTopologySnapshot(
        cells=valid.cells,
        nulls=(
            GeometryNull(
                null_id="wrong-edge",
                axial_position_m=0.5,
                upstream_cell_id=valid.cells[0].cell_id,
                downstream_cell_id=valid.cells[2].cell_id,
                classification=NullClassification.INTERIOR_CUSP,
                loss_probability=valid.interior_cusps[0].loss_probability,
                mirror_ratio=None,
                exclusion_reason=None,
                confidence=1.0,
                provenance_sha256=provenance_hash("wrong-edge"),
            ),
            GeometryNull(
                null_id="duplicate-edge",
                axial_position_m=1.5,
                upstream_cell_id=valid.cells[0].cell_id,
                downstream_cell_id=valid.cells[2].cell_id,
                classification=NullClassification.INTERIOR_CUSP,
                loss_probability=valid.interior_cusps[1].loss_probability,
                mirror_ratio=None,
                exclusion_reason=None,
                confidence=1.0,
                provenance_sha256=provenance_hash("duplicate-edge"),
            ),
        ),
        terminals=(valid.cathode, valid.anode),
        loss_probability_covariance=valid.loss_probability_covariance,
        semantic_hashes=valid.semantic_hashes,
    )
    with pytest.raises(NetworkValidationError, match="adjacent"):
        build_chain_topology(SnapshotAdapter(snapshot))

    with pytest.raises(NetworkValidationError, match="N-1"):
        make_chain_topology(3, (0.1,), provenance_seed="missing-edge")

    wrong_terminal_snapshot = replace(
        snapshot,
        nulls=tuple(
            GeometryNull(
                null_id=cusp.cusp_id,
                axial_position_m=cusp.axial_position_m,
                upstream_cell_id=cusp.upstream_cell_id,
                downstream_cell_id=cusp.downstream_cell_id,
                classification=NullClassification.INTERIOR_CUSP,
                loss_probability=cusp.loss_probability,
                mirror_ratio=cusp.mirror_ratio,
                exclusion_reason=None,
                confidence=cusp.confidence,
                provenance_sha256=cusp.provenance_sha256,
            )
            for cusp in valid.interior_cusps
        ),
        terminals=(
            replace(valid.cathode, adjacent_cell_id=valid.cells[1].cell_id),
            valid.anode,
        ),
    )
    with pytest.raises(NetworkValidationError, match="cathode"):
        build_chain_topology(SnapshotAdapter(wrong_terminal_snapshot))


def test_uncertainty_and_provenance_are_explicit_and_validated() -> None:
    topology = make_chain_topology(
        2,
        (0.15,),
        provenance_seed="uncertainty",
        loss_standard_uncertainties=(0.01,),
        mirror_ratios=(2.5,),
        mirror_standard_uncertainties=(0.2,),
    )
    cusp = topology.interior_cusps[0]
    assert cusp.loss_probability.standard_uncertainty == 0.01
    assert cusp.mirror_ratio is not None
    assert cusp.mirror_ratio.value == 2.5
    with pytest.raises(NetworkValidationError, match="SHA-256"):
        replace(cusp.loss_probability, provenance_sha256="not-a-hash")


def test_semantic_identity_hashes_every_topology_input() -> None:
    topology = make_chain_topology(
        3,
        (0.15, 0.25),
        provenance_seed="identity-completeness",
        loss_standard_uncertainties=(0.01, 0.02),
        mirror_ratios=(2.0, 3.0),
        mirror_standard_uncertainties=(0.1, 0.2),
        finite_boundary_null_count=1,
    )
    base = topology.identity_sha256
    cell = topology.cells[0]
    cusp = topology.interior_cusps[0]
    excluded = topology.excluded_boundary_nulls[0]
    assert cusp.mirror_ratio is not None

    def changed_cell(**changes):
        return replace(
            topology,
            cells=(replace(cell, **changes), *topology.cells[1:]),
        )

    def changed_cusp(**changes):
        return replace(
            topology,
            interior_cusps=(replace(cusp, **changes), *topology.interior_cusps[1:]),
        )

    mutations = (
        changed_cell(axial_position_m=0.05),
        changed_cell(volume_m3=1.1),
        changed_cell(confidence=0.9),
        changed_cell(provenance_sha256=provenance_hash("changed-cell")),
        changed_cusp(cusp_id="changed-cusp"),
        changed_cusp(axial_position_m=0.4),
        changed_cusp(confidence=0.9),
        changed_cusp(provenance_sha256=provenance_hash("changed-cusp-provenance")),
        changed_cusp(
            loss_probability=replace(cusp.loss_probability, value=0.16)
        ),
        changed_cusp(
            loss_probability=replace(
                cusp.loss_probability, standard_uncertainty=0.011
            )
        ),
        changed_cusp(
            loss_probability=replace(
                cusp.loss_probability,
                provenance_sha256=provenance_hash("changed-loss-provenance"),
            )
        ),
        changed_cusp(mirror_ratio=replace(cusp.mirror_ratio, value=2.1)),
        changed_cusp(
            mirror_ratio=replace(cusp.mirror_ratio, standard_uncertainty=0.11)
        ),
        changed_cusp(
            mirror_ratio=replace(
                cusp.mirror_ratio,
                provenance_sha256=provenance_hash("changed-mirror-provenance"),
            )
        ),
        replace(
            topology,
            loss_probability_covariance=((0.0002, 0.0), (0.0, 0.0004)),
        ),
        replace(
            topology,
            excluded_boundary_nulls=(replace(excluded, null_id="changed-null"),),
        ),
        replace(
            topology,
            excluded_boundary_nulls=(
                replace(excluded, axial_position_m=3.1),
            ),
        ),
        replace(
            topology,
            excluded_boundary_nulls=(replace(excluded, reason="alternate reason"),),
        ),
        replace(
            topology,
            excluded_boundary_nulls=(replace(excluded, confidence=0.9),),
        ),
        replace(
            topology,
            excluded_boundary_nulls=(
                replace(
                    excluded,
                    provenance_sha256=provenance_hash("changed-excluded-provenance"),
                ),
            ),
        ),
        replace(topology, cathode=replace(topology.cathode, boundary_id="new-cathode")),
        replace(topology, cathode=replace(topology.cathode, axial_position_m=-0.6)),
        replace(topology, cathode=replace(topology.cathode, confidence=0.9)),
        replace(
            topology,
            cathode=replace(
                topology.cathode,
                provenance_sha256=provenance_hash("changed-cathode-provenance"),
            ),
        ),
    )
    assert all(item.identity_sha256 != base for item in mutations)
    for hash_name in topology.semantic_hashes.__dataclass_fields__:
        changed_hashes = replace(
            topology.semantic_hashes,
            **{hash_name: provenance_hash(f"changed:{hash_name}")},
        )
        assert replace(topology, semantic_hashes=changed_hashes).identity_sha256 != base


def test_direct_construction_and_replace_cannot_bypass_topology_validation() -> None:
    topology = make_chain_topology(3, (0.1, 0.2), provenance_seed="immutable")
    with pytest.raises(NetworkValidationError, match="N-1"):
        replace(topology, interior_cusps=())
    with pytest.raises(NetworkValidationError, match="globally unique"):
        replace(
            topology,
            interior_cusps=(
                replace(
                    topology.interior_cusps[0],
                    cusp_id=topology.cathode.boundary_id,
                ),
                topology.interior_cusps[1],
            ),
        )
    with pytest.raises(NetworkValidationError, match="position"):
        replace(
            topology,
            interior_cusps=(
                replace(topology.interior_cusps[0], axial_position_m=5.0),
                topology.interior_cusps[1],
            ),
        )
    object.__setattr__(topology, "interior_cusps", ())
    with pytest.raises(NetworkValidationError, match="N-1"):
        validate_topology(topology)
    with pytest.raises(NetworkValidationError, match="N-1"):
        generate_equation_ledger(topology)

    case = manufactured_zero_cusp_case(2)
    object.__setattr__(case.inputs.topology, "interior_cusps", ())
    with pytest.raises(NetworkValidationError, match="N-1"):
        replace(case.inputs, topology=case.inputs.topology)
    with pytest.raises(NetworkValidationError, match="N-1"):
        solve_network(case.inputs, case.state)
