from dataclasses import replace

import pytest

from cft_revival.geometry import (
    GeometryAdapter,
    GeometryValidationError,
    MaterialKind,
    PermanentMagnetAuthority,
    PermanentMagnetRepresentationPlan,
    compact_high_gradient_stack,
    divergent_exit_stack,
    interface_topology,
    to_l1a_current_equivalent_preview,
    to_magnetics_handoff,
)
from cft_revival.magnetics import (
    ConstitutiveLawKind,
    LinearPermeability,
    PermanentMagnetRepresentation,
    checked_synthetic_smco_like_magnet,
)


def material_registry(geometry):
    registry = {}
    for definition in geometry.materials:
        if definition.category is MaterialKind.PERMANENT_MAGNET:
            material = checked_synthetic_smco_like_magnet()
            assert material.material_id == definition.material_id
        else:
            material = LinearPermeability(
                definition.material_id, definition.relative_permeability
            )
        registry[definition.material_id] = material
    return registry


def equivalent_geometry(geometry):
    authority = PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT
    return replace(
        geometry,
        permanent_magnet_plan=PermanentMagnetRepresentationPlan(
            plan_id=f"{geometry.config_id}-{authority.value}-v1",
            authority=authority,
        ),
    )


def test_solver_neutral_contract_has_complete_oriented_topology() -> None:
    geometry = divergent_exit_stack()
    contract = GeometryAdapter(geometry).solver_neutral_contract()
    decoded = contract.to_dict()
    assert contract.contract_version == "cft_revival.geometry.solver_neutral/1.1.0"
    assert decoded["material_regions"][0]["region_id"] == "ambient-background"
    assert any(
        region["shape"] == "linear_taper_annulus"
        for region in decoded["material_regions"]
    )
    surfaces_by_region = {}
    for interface in decoded["interfaces"]:
        surfaces_by_region.setdefault(interface["region_id"], set()).add(
            interface["surface"]
        )
        radial = interface["unit_normal_rz"]["radial"]
        axial = interface["unit_normal_rz"]["axial"]
        assert radial * radial + axial * axial == pytest.approx(1.0)
    assert all(
        surfaces == {"inner", "outer", "z_min", "z_max"}
        for surfaces in surfaces_by_region.values()
    )
    tapered_outer = next(
        interface
        for interface in interface_topology(geometry)
        if interface.region_id == "channel-divergent-exit"
        and interface.surface == "outer"
    )
    assert tapered_outer.adjacent_region_id == "dielectric-divergent-exit"
    assert tapered_outer.unit_normal_rz == pytest.approx(
        (0.9863939238321437, -0.1643989873053573)
    )
    assert {component["kind"] for component in decoded["external_components"]} == {
        "cathode",
        "neutralizer",
    }


def test_interface_graph_is_deterministic_reciprocal_and_connected() -> None:
    geometry = divergent_exit_stack()
    topology = interface_topology(geometry)
    assert topology == interface_topology(geometry)
    region_ids = {region.region_id for region in geometry.regions}
    graph = {region_id: set() for region_id in region_ids}
    graph["ambient-background"] = set()
    graph["symmetry-axis"] = set()
    for descriptor in topology:
        graph[descriptor.region_id].add(descriptor.adjacent_region_id)
        graph[descriptor.adjacent_region_id].add(descriptor.region_id)
        if descriptor.adjacent_region_id in region_ids:
            reciprocal = [
                candidate
                for candidate in topology
                if candidate.region_id == descriptor.adjacent_region_id
                and candidate.adjacent_region_id == descriptor.region_id
                and candidate.start_rz_m == pytest.approx(descriptor.start_rz_m)
                and candidate.end_rz_m == pytest.approx(descriptor.end_rz_m)
            ]
            assert reciprocal
            assert sum(
                left * right
                for left, right in zip(
                    descriptor.unit_normal_rz, reciprocal[0].unit_normal_rz
                )
            ) == pytest.approx(-1.0)
    pending = ["ambient-background"]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(graph[current] - visited)
    assert region_ids <= visited


def test_recoil_handoff_uses_serialized_plan_and_supplied_registry() -> None:
    geometry = compact_high_gradient_stack()
    contract = to_magnetics_handoff(
        geometry, material_registry=material_registry(geometry)
    )
    magnet_regions = [
        region
        for region in contract.regions
        if region.constitutive_law_kind
        is ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL
    ]
    geometry_pm_count = sum(
        region.role == "permanent_magnet" for region in geometry.regions
    )
    assert len(magnet_regions) == geometry_pm_count == len(geometry.stages)
    assert contract.magnetization_sources == ()
    assert all(
        region.permanent_magnet_representation
        is PermanentMagnetRepresentation.RECOIL_REMANENCE
        for region in magnet_regions
    )
    assert any(
        interface.plus_region_id == "shield-shell"
        for interface in contract.interfaces
    )


def test_equivalent_authority_requires_distinct_bound_representation_plan() -> None:
    recoil = compact_high_gradient_stack()
    with pytest.raises(GeometryValidationError, match="bind config ID and authority"):
        replace(
            recoil,
            permanent_magnet_plan=replace(
                recoil.permanent_magnet_plan,
                authority=PermanentMagnetAuthority.EQUIVALENT_BOUND_CURRENT,
            ),
        )
    geometry = equivalent_geometry(recoil)
    contract = to_magnetics_handoff(
        geometry, material_registry=material_registry(geometry)
    )
    geometry_pm_count = sum(
        region.role == "permanent_magnet" for region in geometry.regions
    )
    assert len(contract.magnetization_sources) == geometry_pm_count
    assert not any(
        region.constitutive_law_kind
        is ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL
        for region in contract.regions
    )
    assert all(
        source.representation is PermanentMagnetRepresentation.EQUIVALENT_BOUND_CURRENT
        for source in contract.magnetization_sources
    )


def test_registry_resolution_never_substitutes_unknown_or_wrong_materials() -> None:
    geometry = compact_high_gradient_stack()
    registry = material_registry(geometry)
    registry.pop("copper-anode-assumed")
    with pytest.raises(GeometryValidationError, match="missing"):
        to_magnetics_handoff(geometry, material_registry=registry)

    registry = material_registry(geometry)
    registry["copper-anode-assumed"] = checked_synthetic_smco_like_magnet()
    with pytest.raises(GeometryValidationError, match="key and material_id"):
        to_magnetics_handoff(geometry, material_registry=registry)

    registry = material_registry(geometry)
    registry["synthetic-smco-like-example-v1"] = LinearPermeability(
        "synthetic-smco-like-example-v1", 1.05
    )
    with pytest.raises(GeometryValidationError, match="must resolve"):
        to_magnetics_handoff(geometry, material_registry=registry)


def test_l1a_preview_is_non_authoritative_typed_output() -> None:
    geometry = compact_high_gradient_stack()
    preview = to_l1a_current_equivalent_preview(
        geometry,
        material_registry=material_registry(geometry),
        radial_smear_thickness_m=0.00025,
    )
    assert preview.authoritative is False
    assert preview.representation_plan_id == geometry.permanent_magnet_plan.plan_id
    assert "not a material-aware" in preview.approximation
    assert len(preview.bands) == 2 * len(geometry.stages)
    for inner, outer in zip(preview.bands[::2], preview.bands[1::2]):
        assert inner.ampere_turns_a == pytest.approx(outer.ampere_turns_a)
        assert inner.polarity == -outer.polarity
        assert "l1a-preview" in inner.name
    with pytest.raises(GeometryValidationError, match="fit inside"):
        to_l1a_current_equivalent_preview(
            geometry,
            material_registry=material_registry(geometry),
            radial_smear_thickness_m=0.002,
        )
