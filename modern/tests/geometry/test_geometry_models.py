import json
from dataclasses import replace

import pytest

from cft_revival.geometry import (
    GeometryValidationError,
    ManufacturingRules,
    MagnetizationDirection,
    canonical_json,
    deserialize_geometry,
    historical_envelope_baseline,
)


def test_reference_geometry_is_si_closed_and_hash_stable() -> None:
    geometry = historical_envelope_baseline()
    assert geometry.length_unit == "m"
    assert geometry.classification == "hypothetical_not_optimized_not_build_qualified"
    assert len(geometry.canonical_sha256) == 64
    serialized = canonical_json(geometry.to_dict())
    assert deserialize_geometry(serialized) == geometry
    assert canonical_json(geometry.to_dict()) == serialized
    decoded = json.loads(serialized)
    assert decoded["integrity"]["payload_sha256"] == geometry.canonical_sha256


def test_closed_constructor_rejects_extra_key_tamper_and_noncanonical_json() -> None:
    geometry = historical_envelope_baseline()
    decoded = geometry.to_dict()
    decoded["unexpected"] = True
    with pytest.raises(GeometryValidationError, match="extra"):
        deserialize_geometry(canonical_json(decoded))

    decoded = geometry.to_dict()
    decoded["chamber"]["length_m"] = 0.022
    with pytest.raises(GeometryValidationError, match="coverage|SHA-256"):
        deserialize_geometry(canonical_json(decoded))

    pretty = json.dumps(geometry.to_dict(), sort_keys=True)
    with pytest.raises(GeometryValidationError, match="canonical"):
        deserialize_geometry(pretty)


def test_duplicate_keys_nonfinite_and_bad_units_are_rejected() -> None:
    with pytest.raises(GeometryValidationError, match="duplicate"):
        deserialize_geometry('{"schema_version":"x","schema_version":"y"}')
    geometry = historical_envelope_baseline()
    decoded = geometry.to_dict()
    decoded["chamber"]["length_m"] = float("nan")
    with pytest.raises(GeometryValidationError, match="finite"):
        deserialize_geometry(canonical_json(decoded))
    with pytest.raises(GeometryValidationError, match="SI metres"):
        replace(geometry, length_unit="mm")


def test_overlap_ordering_and_alternating_polarity_are_enforced() -> None:
    geometry = historical_envelope_baseline()
    probe = replace(
        geometry.region_by_id("channel-straight"),
        region_id="overlap-probe",
        owner_id="overlap-probe",
    )
    regions = tuple(
        sorted(
            (*geometry.regions, probe),
            key=lambda region: (
                region.z_min_m,
                region.r_inner_start_m,
                region.r_inner_end_m,
                region.region_id,
            ),
        )
    )
    with pytest.raises(GeometryValidationError, match="overlap"):
        replace(geometry, regions=regions)
    with pytest.raises(GeometryValidationError, match="geometric ordering"):
        replace(geometry, regions=tuple(reversed(geometry.regions)))

    stage_2 = replace(
        geometry.stages[1],
        magnetization=MagnetizationDirection.AXIAL_POSITIVE,
    )
    magnet_2 = replace(geometry.region_by_id("magnet-02"), polarity=1)
    altered_regions = tuple(
        magnet_2 if region.region_id == magnet_2.region_id else region
        for region in geometry.regions
    )
    with pytest.raises(GeometryValidationError, match="alternate"):
        replace(
            geometry,
            stages=(geometry.stages[0], stage_2, geometry.stages[2]),
            regions=altered_regions,
        )


def test_manufacturing_tolerance_and_clearance_rules_are_strict() -> None:
    with pytest.raises(GeometryValidationError, match="tolerance stack"):
        ManufacturingRules(2.0e-4, 1.0e-4, 1.0e-4, 2.0e-5, 1.0e-4)
    with pytest.raises(GeometryValidationError, match="at least"):
        ManufacturingRules(3.0e-4, 2.0e-4, 2.0e-5, 2.0e-5, 1.0e-4)

    geometry = historical_envelope_baseline()
    chamber = replace(geometry.chamber, dielectric_thickness_m=2.0e-4)
    with pytest.raises(GeometryValidationError, match="dielectric wall"):
        replace(geometry, chamber=chamber)


def test_region_references_axis_regularity_and_external_metadata() -> None:
    geometry = historical_envelope_baseline()
    assert geometry.chamber.inner_radius_m == 0.0
    assert all(
        geometry.region_by_id(stage.magnet_region_id).r_inner_start_m > 0.0
        for stage in geometry.stages
    )
    external = {component.kind: component for component in geometry.external_components}
    assert set(external) == {"cathode", "neutralizer"}
    assert all(not component.included_in_2d_model for component in external.values())
    bad = replace(geometry.regions[0], material_id="missing-material")
    with pytest.raises(GeometryValidationError, match="supplied material"):
        replace(geometry, regions=(bad, *geometry.regions[1:]))
