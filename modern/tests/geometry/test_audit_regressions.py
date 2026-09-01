from dataclasses import replace
from math import nextafter

import pytest

from cft_revival.geometry import (
    GeometryValidationError,
    MaterialDefinition,
    MaterialKind,
    MeridionalRegion,
    RegionShape,
    canonical_json,
    compact_high_gradient_stack,
    divergent_exit_stack,
    svg_meridional_cross_section,
)
from cft_revival.geometry.descriptors import _representable_mass_kg


def sorted_regions(regions):
    return tuple(
        sorted(
            regions,
            key=lambda region: (
                region.z_min_m,
                region.r_inner_start_m,
                region.r_inner_end_m,
                region.region_id,
            ),
        )
    )


def test_stage_centers_references_and_region_kinds_are_closed() -> None:
    geometry = compact_high_gradient_stack()
    with pytest.raises(GeometryValidationError, match="center"):
        replace(
            geometry,
            stages=(
                replace(geometry.stages[0], center_z_m=0.002000001),
                *geometry.stages[1:],
            ),
        )
    duplicate = replace(
        geometry.stages[1],
        magnet_region_id=geometry.stages[0].magnet_region_id,
        magnetization=geometry.stages[0].magnetization,
    )
    with pytest.raises(GeometryValidationError, match="references must be unique"):
        replace(
            geometry,
            stages=(geometry.stages[0], duplicate, *geometry.stages[2:]),
        )
    pole = geometry.region_by_id("pole-01")
    wrong_pole = replace(pole, role="shield", material_id="al6061-assumed")
    with pytest.raises(GeometryValidationError, match="pole-piece"):
        replace(
            geometry,
            regions=tuple(
                wrong_pole if region.region_id == pole.region_id else region
                for region in geometry.regions
            ),
        )
    with pytest.raises(GeometryValidationError, match="anode"):
        replace(
            geometry,
            electrodes=replace(
                geometry.electrodes, anode_region_id="injector-zone"
            ),
        )


def test_stage_envelopes_and_pole_adjacency_are_model_invariants() -> None:
    geometry = compact_high_gradient_stack()
    first_stage = geometry.stages[0]
    with pytest.raises(GeometryValidationError, match="outside stage/chamber"):
        replace(
            geometry,
            stages=(
                replace(
                    first_stage,
                    z_min_m=nextafter(
                        geometry.region_by_id(first_stage.magnet_region_id).z_min_m,
                        float("inf"),
                    ),
                ),
                *geometry.stages[1:],
            ),
        )

    first_pole = geometry.region_by_id(first_stage.pole_after_region_id)
    separated_pole = replace(
        first_pole,
        z_min_m=nextafter(first_pole.z_min_m, float("inf")),
    )
    with pytest.raises(GeometryValidationError, match="pole_after"):
        replace(
            geometry,
            regions=sorted_regions(
                tuple(
                    separated_pole
                    if region.region_id == first_pole.region_id
                    else region
                    for region in geometry.regions
                )
            ),
        )

    shift = geometry.chamber.length_m
    shifted_region_ids = {
        stage.magnet_region_id for stage in geometry.stages
    } | {
        stage.pole_after_region_id
        for stage in geometry.stages
        if stage.pole_after_region_id is not None
    }
    shifted_regions = sorted_regions(
        tuple(
            replace(
                region,
                z_min_m=region.z_min_m + shift,
                z_max_m=region.z_max_m + shift,
            )
            if region.region_id in shifted_region_ids
            else region
            for region in geometry.regions
        )
    )
    shifted_stages = tuple(
        replace(
            stage,
            center_z_m=stage.center_z_m + shift,
            z_min_m=stage.z_min_m + shift,
            z_max_m=stage.z_max_m + shift,
        )
        for stage in geometry.stages
    )
    with pytest.raises(GeometryValidationError, match="outside chamber"):
        replace(geometry, regions=shifted_regions, stages=shifted_stages)


def test_nonrectangular_permanent_magnets_are_rejected_before_handoff() -> None:
    geometry = divergent_exit_stack()
    magnet = geometry.region_by_id(geometry.stages[-1].magnet_region_id)
    tapered = replace(
        magnet,
        shape=RegionShape.LINEAR_TAPER_ANNULUS,
        r_inner_end_m=magnet.r_inner_end_m - 0.0002,
    )
    with pytest.raises(GeometryValidationError, match="must be rectangular"):
        replace(
            geometry,
            regions=sorted_regions(
                tuple(
                    tapered if region.region_id == magnet.region_id else region
                    for region in geometry.regions
                )
            ),
        )


def test_chamber_coverage_and_divergent_continuity_reject_hidden_discontinuity() -> None:
    geometry = divergent_exit_stack()
    straight = geometry.region_by_id("channel-straight")
    gap_region = replace(
        straight,
        z_max_m=nextafter(straight.z_max_m, float("-inf")),
    )
    with pytest.raises(GeometryValidationError, match="axial gap"):
        replace(
            geometry,
            regions=sorted_regions(
                tuple(
                    gap_region if region.region_id == straight.region_id else region
                    for region in geometry.regions
                )
            ),
        )

    wall = geometry.region_by_id("dielectric-divergent-exit")
    altered_radius = wall.r_outer_end_m
    for _ in range(8):
        altered_radius = nextafter(altered_radius, float("inf"))
    discontinuous = replace(wall, r_outer_end_m=altered_radius)
    with pytest.raises(GeometryValidationError, match="ULP-continuous|slopes"):
        replace(
            geometry,
            regions=sorted_regions(
                tuple(
                    discontinuous if region.region_id == wall.region_id else region
                    for region in geometry.regions
                )
            ),
        )


def test_clearance_touch_one_ulp_and_shortfall_fail_without_rounding() -> None:
    geometry = compact_high_gradient_stack()
    dielectric_outer = (
        geometry.chamber.outer_radius_m + geometry.chamber.dielectric_thickness_m
    )
    required = (
        geometry.manufacturing.thermal_clearance_m
        + 2.0 * geometry.manufacturing.radial_tolerance_m
    )
    for magnet_inner in (
        dielectric_outer,
        nextafter(dielectric_outer, float("inf")),
        nextafter(dielectric_outer + required, float("-inf")),
    ):
        changed = tuple(
            replace(
                region,
                r_inner_start_m=magnet_inner,
                r_inner_end_m=magnet_inner,
            )
            if region.role in ("permanent_magnet", "pole_piece")
            else region
            for region in geometry.regions
        )
        with pytest.raises(GeometryValidationError, match="clearance"):
            replace(geometry, regions=sorted_regions(changed))


def test_density_volume_and_mass_publication_domains_are_checked() -> None:
    with pytest.raises(GeometryValidationError, match="density"):
        MaterialDefinition(
            "bad-density",
            MaterialKind.PERMANENT_MAGNET,
            1.05,
            1.0e-4,
            "adversarial test",
            True,
        )
    huge = MeridionalRegion(
        "huge-region",
        "huge-owner",
        "shield",
        "al6061-assumed",
        RegionShape.RECTANGULAR_ANNULUS,
        1.0e200,
        1.0e200,
        2.0e200,
        2.0e200,
        0.0,
        1.0e200,
    )
    with pytest.raises(GeometryValidationError, match="volume"):
        _ = huge.volume_m3
    with pytest.raises(GeometryValidationError, match="underflowed"):
        _representable_mass_kg(float.fromhex("0x0.0000000000010p-1022"), 1.0e-3)


@pytest.mark.parametrize(
    "unsafe",
    ("../escape", "a/b", "a\\b", 'a"b', "a\x01b", "..", " leading"),
)
def test_unsafe_identifiers_and_filename_traversal_are_rejected(unsafe: str) -> None:
    geometry = compact_high_gradient_stack()
    with pytest.raises(GeometryValidationError, match="safe canonical"):
        replace(geometry, config_id=unsafe)


def test_svg_dynamic_text_and_attributes_are_xml_escaped() -> None:
    geometry = replace(
        compact_high_gradient_stack(),
        title='"><script>alert("geometry")</script>&',
    )
    svg = svg_meridional_cross_section(geometry)
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg
    assert "&amp;" in svg
    assert canonical_json(geometry.to_dict()).startswith("{")
