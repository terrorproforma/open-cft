from pathlib import Path

import pytest

from cft_revival.geometry import (
    compact_high_gradient_stack,
    compute_descriptors,
    divergent_exit_stack,
    geometry_with_design_vector,
    historical_envelope_baseline,
    reference_variants,
)


def test_historical_variant_maps_only_traceable_fyp_dimensions() -> None:
    geometry = historical_envelope_baseline()
    assert geometry.chamber.outer_radius_m == pytest.approx(0.002)
    assert geometry.chamber.length_m == pytest.approx(0.021)
    spans = [
        (
            geometry.region_by_id(stage.magnet_region_id).z_min_m,
            geometry.region_by_id(stage.magnet_region_id).z_max_m,
        )
        for stage in geometry.stages
    ]
    expected = ((0.0, 0.004), (0.005, 0.015), (0.016, 0.020))
    for actual_span, expected_span in zip(spans, expected):
        assert actual_span == pytest.approx(expected_span)
    evidence = {note.note_id: note for note in geometry.evidence}
    assert evidence["legacy-fixed-envelope"].classification == "traceable"
    assert evidence["historical-clearance-assumption"].classification == "assumption"
    assert "not CFT plasma" in evidence["twt-physics-boundary"].statement

    repository = Path(__file__).resolve().parents[3]
    source = (repository / "FYP" / "FEMMrun.m").read_text(encoding="utf-8")
    for literal in ("a = 2;", "f = 4;", "g = 5;", "h = 15;", "k = 16;", "l = 20;", "p = 21;"):
        assert literal in source


def test_three_variants_are_distinct_and_explicitly_hypothetical() -> None:
    variants = reference_variants()
    assert len(variants) == 3
    assert len({variant.config_id for variant in variants}) == 3
    assert len({variant.canonical_sha256 for variant in variants}) == 3
    assert all(
        variant.classification == "hypothetical_not_optimized_not_build_qualified"
        for variant in variants
    )
    assert len(compact_high_gradient_stack().stages) == 5
    divergent = divergent_exit_stack()
    assert divergent.chamber.exit_length_m == pytest.approx(0.006)
    assert divergent.chamber.exit_outer_radius_m == pytest.approx(0.003)
    assert {
        region.shape.value
        for region in divergent.regions
        if "divergent" in region.region_id
    } == {"linear_taper_annulus"}


@pytest.mark.parametrize("factory", reference_variants())
def test_descriptors_are_geometry_only_and_vector_mapping_is_stable(factory) -> None:
    descriptors = compute_descriptors(factory)
    assert descriptors.active_volume_m3 > descriptors.channel_volume_m3 > 0.0
    assert descriptors.channel_exit_area_m2 >= descriptors.channel_inlet_area_m2
    assert descriptors.cusp_count == len(factory.stages) - 1
    assert descriptors.minimum_radial_gap_m >= factory.manufacturing.thermal_clearance_m
    exact_gaps = []
    walls = [
        region for region in factory.regions if region.role == "dielectric_wall"
    ]
    for stage in factory.stages:
        magnet = factory.region_by_id(stage.magnet_region_id)
        for wall in walls:
            z_min = max(magnet.z_min_m, wall.z_min_m)
            z_max = min(magnet.z_max_m, wall.z_max_m)
            if z_max > z_min:
                exact_gaps.extend(
                    (
                        magnet.r_inner_start_m
                        - wall.radial_interval_at(z_min)[1],
                        magnet.r_inner_end_m
                        - wall.radial_interval_at(z_max)[1],
                    )
                )
    assert descriptors.minimum_radial_gap_m == min(exact_gaps)
    assert descriptors.minimum_axial_gap_m > 0.0
    assert descriptors.magnet_mass_estimate_kg > 0.0
    assert descriptors.envelope_radius_m > factory.chamber.outer_radius_m
    assert descriptors.manufacturability_warnings
    assert descriptors.design_variable_names == factory.design_variable_order
    mapping = geometry_with_design_vector(
        factory, descriptors.design_variable_values_si
    )
    assert tuple(mapping) == factory.design_variable_order
    assert "performance" in descriptors.to_dict()["claim_limit"]
