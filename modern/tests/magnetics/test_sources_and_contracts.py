import json
import math

import pytest

from cft_revival.magnetics import (
    MU0_H_PER_M,
    AxisymmetricBoundCurrentSheet,
    AxisymmetricBounds,
    AxisymmetricMaterialProblemContract,
    AxisymmetricTruncationDomain,
    ConstitutiveLawKind,
    LinearPermeability,
    MagneticsValidationError,
    MaterialInterfaceContract,
    MaterialRegionContract,
    OpenBoundaryDomainPolicy,
    PermanentMagnetRepresentation,
    SheetOrientation,
    SmCoPermanentMagnet,
    UniformAxisymmetricMagnetizationSource,
    VectorRZ,
    WarningSeverity,
    assess_demagnetization,
    bound_surface_current_density_phi_a_per_m,
    bound_volume_current_density_phi_a_per_m2,
    canonical_json,
    checked_synthetic_smco_like_magnet,
    checked_synthetic_soft_magnetic_curve,
)


def test_bound_current_equations_and_uniform_volume_limit() -> None:
    magnetization = VectorRZ(20.0, 30.0)
    assert bound_volume_current_density_phi_a_per_m2(7.0, 2.0) == 5.0
    assert bound_surface_current_density_phi_a_per_m(
        magnetization, VectorRZ(1.0, 0.0)
    ) == 30.0
    assert bound_surface_current_density_phi_a_per_m(
        magnetization, VectorRZ(0.0, 1.0)
    ) == -20.0

    material = SmCoPermanentMagnet(
        "material-1",
        MU0_H_PER_M * magnetization.magnitude,
        1.0e6,
        1.05,
        300.0,
        0.0,
        0.0,
        290.0,
        310.0,
        "test synthetic",
        True,
    )
    source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="magnet-1",
        region_id="magnet-host",
        material=material,
        bounds=AxisymmetricBounds(0.01, 0.02, -0.03, 0.04),
        direction=magnetization,
        temperature_k=300.0,
    )
    sheets = {sheet.surface_name: sheet for sheet in source.equivalent_bound_current_sheets()}
    assert source.bound_volume_current_density_phi_a_per_m2 == 0.0
    assert sheets["r_inner"].k_phi_a_per_m == pytest.approx(-30.0)
    assert sheets["r_outer"].k_phi_a_per_m == pytest.approx(30.0)
    assert sheets["z_min"].k_phi_a_per_m == pytest.approx(20.0)
    assert sheets["z_max"].k_phi_a_per_m == pytest.approx(-20.0)


def test_axis_touching_magnet_omits_degenerate_inner_surface() -> None:
    material = checked_synthetic_smco_like_magnet()
    source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="solid-magnet",
        region_id="solid-host",
        material=material,
        bounds=AxisymmetricBounds(0.0, 0.02, -0.01, 0.01),
        direction=VectorRZ(0.0, 1.0),
        temperature_k=material.reference_temperature_k,
    )
    names = tuple(sheet.surface_name for sheet in source.equivalent_bound_current_sheets())
    assert names == ("r_outer", "z_min", "z_max")
    radial_sheet = source.equivalent_bound_current_sheets()[0]
    assert radial_sheet.coordinate_m == source.bounds.r_outer_m


def test_axis_regularity_and_current_sheet_geometry_are_enforced() -> None:
    material = checked_synthetic_smco_like_magnet()
    with pytest.raises(MagneticsValidationError, match="radial magnetization"):
        UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
            source_id="irregular",
            region_id="host",
            material=material,
            bounds=AxisymmetricBounds(0.0, 0.02, -0.01, 0.01),
            direction=VectorRZ(1.0, 1.0),
            temperature_k=material.reference_temperature_k,
        )
    with pytest.raises(MagneticsValidationError, match="radius must be positive"):
        AxisymmetricBoundCurrentSheet(
            "source",
            "axis",
            SheetOrientation.CONSTANT_R,
            0.0,
            -1.0,
            1.0,
            VectorRZ(1.0, 0.0),
            1.0,
        )
    with pytest.raises(MagneticsValidationError, match="radial span"):
        AxisymmetricBoundCurrentSheet(
            "source",
            "plane",
            SheetOrientation.CONSTANT_Z,
            0.0,
            -1.0,
            1.0,
            VectorRZ(0.0, 1.0),
            1.0,
        )
    with pytest.raises(MagneticsValidationError, match="normal must be radial"):
        AxisymmetricBoundCurrentSheet(
            "source",
            "angled",
            SheetOrientation.CONSTANT_R,
            1.0,
            -1.0,
            1.0,
            VectorRZ(1.0, 1.0e-12),
            1.0,
        )
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    with pytest.raises(MagneticsValidationError, match="axial thickness"):
        AxisymmetricBounds(0.0, 1.0, -maximum, maximum)


def test_vector_normalization_is_overflow_safe_and_magnitude_is_typed() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    direction = VectorRZ(maximum, maximum).normalized()
    expected = 2.0**-0.5
    assert direction.radial == pytest.approx(expected)
    assert direction.axial == pytest.approx(expected)
    with pytest.raises(MagneticsValidationError, match="vector magnitude"):
        _ = VectorRZ(maximum, maximum).magnitude
    with pytest.raises(MagneticsValidationError, match="zero vector"):
        MaterialInterfaceContract("bad-normal", "a", "b", VectorRZ(0.0, -0.0))


def test_permanent_magnet_source_uses_temperature_adjusted_remanence() -> None:
    material = checked_synthetic_smco_like_magnet()
    source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="pm",
        region_id="pm-host-region",
        material=material,
        bounds=AxisymmetricBounds(0.01, 0.02, -0.02, 0.02),
        direction=VectorRZ(0.0, 1.0),
        temperature_k=373.15,
    )
    assert source.temperature_k == 373.15
    assert source.magnetization_a_per_m.axial > 0.0
    assert source.magnetization_a_per_m.radial == 0.0
    assert source.magnetization_a_per_m.magnitude == pytest.approx(
        material.remanence_t(373.15) / MU0_H_PER_M
    )
    with pytest.raises(MagneticsValidationError, match="outside"):
        UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
            source_id="invalid-temperature",
            region_id="pm-host-region",
            material=material,
            bounds=AxisymmetricBounds(0.01, 0.02, -0.02, 0.02),
            direction=VectorRZ(0.0, 1.0),
            temperature_k=1.0,
        )


def test_interface_contract_residuals_follow_maxwell_jump_signs() -> None:
    interface = MaterialInterfaceContract(
        "iron-air",
        "iron",
        "air",
        VectorRZ(1.0, 0.0),
        free_surface_current_phi_a_per_m=5.0,
    )
    normal_b, tangential_h = interface.residuals(
        b_minus_t=VectorRZ(0.3, 0.2),
        b_plus_t=VectorRZ(0.3, -0.7),
        h_minus_a_per_m=VectorRZ(10.0, 20.0),
        h_plus_a_per_m=VectorRZ(-5.0, 15.0),
    )
    assert normal_b == pytest.approx(0.0)
    assert tangential_h == pytest.approx(0.0)
    assert "must not be inserted as free current" in interface.__doc__


def test_demagnetization_screen_reports_safe_warning_invalid_and_temperature_unknown() -> None:
    material = checked_synthetic_smco_like_magnet()
    temperature = material.reference_temperature_k
    direction = VectorRZ(0.0, 1.0)
    hci = material.intrinsic_coercivity_a_per_m(temperature)

    safe = assess_demagnetization(
        material=material,
        temperature_k=temperature,
        magnetization_direction=direction,
        local_h_a_per_m=VectorRZ(0.0, -0.5 * hci),
    )
    assert safe.status == "within_screening_limit"
    assert not safe.warnings

    warning = assess_demagnetization(
        material=material,
        temperature_k=temperature,
        magnetization_direction=direction,
        local_h_a_per_m=VectorRZ(0.0, -0.9 * hci),
    )
    assert warning.status == "warning"
    assert warning.warnings[0].code == "demagnetization_margin_low"

    invalid = assess_demagnetization(
        material=material,
        temperature_k=temperature,
        magnetization_direction=direction,
        local_h_a_per_m=VectorRZ(0.0, -1.01 * hci),
    )
    assert invalid.status == "invalid"
    assert invalid.warnings[0].severity is WarningSeverity.INVALID

    unknown = assess_demagnetization(
        material=material,
        temperature_k=material.valid_temperature_max_k + 1.0,
        magnetization_direction=direction,
        local_h_a_per_m=VectorRZ(0.0, 0.0),
    )
    assert unknown.status == "indeterminate"
    assert unknown.intrinsic_coercivity_a_per_m is None
    assert unknown.warnings[0].code == "temperature_outside_material_validity"
    assert "screening check" in unknown.limitations[0]


def test_open_boundary_policy_accepts_only_padded_converged_domain() -> None:
    source_bounds = (AxisymmetricBounds(0.01, 0.02, -0.02, 0.02),)
    policy = OpenBoundaryDomainPolicy()
    accepted = policy.assess(
        domain=AxisymmetricTruncationDomain(0.2, -0.2, 0.2),
        source_bounds=source_bounds,
        maximum_boundary_field_t=5.0e-5,
        maximum_interior_field_t=1.0,
        domain_expansion_factors=(1.5, 1.6),
        qoi_relative_changes_on_expansion=(8.0e-4, 3.0e-4),
    )
    assert accepted == ()

    warnings = policy.assess(
        domain=AxisymmetricTruncationDomain(0.05, -0.05, 0.05),
        source_bounds=source_bounds,
        maximum_boundary_field_t=0.02,
        maximum_interior_field_t=1.0,
        domain_expansion_factors=(1.5,),
        qoi_relative_changes_on_expansion=(2.0e-3,),
    )
    codes = {warning.code for warning in warnings}
    assert codes == {
        "open_boundary_padding_insufficient",
        "boundary_field_not_negligible",
        "domain_expansion_evidence_missing",
    }
    assert "not an exact" in policy.to_dict()["claim_limit"]

    weak_expansions = policy.assess(
        domain=AxisymmetricTruncationDomain(0.2, -0.2, 0.2),
        source_bounds=source_bounds,
        maximum_boundary_field_t=0.0,
        maximum_interior_field_t=1.0,
        domain_expansion_factors=(1.5, 1.1),
        qoi_relative_changes_on_expansion=(0.0, 0.0),
    )
    assert {warning.code for warning in weak_expansions} == {
        "domain_expansion_factor_insufficient"
    }


def test_open_boundary_and_source_extremes_are_rejected() -> None:
    policy = OpenBoundaryDomainPolicy()
    source_bounds = (AxisymmetricBounds(0.01, 0.02, -0.02, 0.02),)
    with pytest.raises(MagneticsValidationError):
        policy.assess(
            domain=AxisymmetricTruncationDomain(0.1, -0.1, 0.1),
            source_bounds=source_bounds,
            maximum_boundary_field_t=math.inf,
            maximum_interior_field_t=1.0,
            domain_expansion_factors=(1.5, 1.5),
            qoi_relative_changes_on_expansion=(0.0, 0.0),
        )
    with pytest.raises(MagneticsValidationError, match="strictly inside"):
        policy.assess(
            domain=AxisymmetricTruncationDomain(0.02, -0.1, 0.1),
            source_bounds=source_bounds,
            maximum_boundary_field_t=0.0,
            maximum_interior_field_t=1.0,
            domain_expansion_factors=(1.5, 1.5),
            qoi_relative_changes_on_expansion=(0.0, 0.0),
        )
    with pytest.raises(MagneticsValidationError):
        AxisymmetricBounds(-0.1, 0.2, 0.0, 1.0)
    with pytest.raises(MagneticsValidationError):
        bound_volume_current_density_phi_a_per_m2(math.nan, 0.0)
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    with pytest.raises(MagneticsValidationError, match="bound volume"):
        bound_volume_current_density_phi_a_per_m2(maximum, -maximum)


def test_solver_handoff_is_self_contained_validated_and_deterministic() -> None:
    material = checked_synthetic_smco_like_magnet()
    air = LinearPermeability("air", 1.0)
    magnet_host = LinearPermeability(
        "pm-recoil-host", material.recoil_relative_permeability
    )
    magnet_region_bounds = AxisymmetricBounds(0.01, 0.02, -0.02, 0.02)
    air_bounds = AxisymmetricBounds(0.0, 0.2, -0.2, 0.2)
    source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="pm",
        region_id="pm-region",
        material=material,
        bounds=magnet_region_bounds,
        direction=VectorRZ(0.0, 1.0),
        temperature_k=material.reference_temperature_k,
    )
    contract = AxisymmetricMaterialProblemContract(
        problem_id="handoff-example",
        materials=(material, magnet_host, air),
        regions=(
            MaterialRegionContract(
                "air-region",
                "air",
                ConstitutiveLawKind.LINEAR_ISOTROPIC,
                air_bounds,
                priority=0,
            ),
            MaterialRegionContract(
                "pm-region",
                magnet_host.material_id,
                ConstitutiveLawKind.LINEAR_ISOTROPIC,
                magnet_region_bounds,
                priority=10,
            ),
        ),
        interfaces=(
            MaterialInterfaceContract(
                "pm-air-outer",
                "pm-region",
                "air-region",
                VectorRZ(1.0, 0.0),
            ),
        ),
        magnetization_sources=(source,),
        open_boundary_policy=OpenBoundaryDomainPolicy(),
    )
    serialized = canonical_json(contract)
    parsed = json.loads(serialized)
    assert canonical_json(contract) == serialized
    assert parsed["contract_version"] == "1.0.0"
    assert parsed["coordinate_system"] == "right_handed_cylindrical_r_phi_z"
    assert "double-count" in parsed["solver_requirements"]["current_semantics"]


def test_solver_handoff_rejects_dangling_material_and_region_references() -> None:
    curve = checked_synthetic_soft_magnetic_curve()
    bounds = AxisymmetricBounds(0.0, 0.1, -0.1, 0.1)
    with pytest.raises(MagneticsValidationError, match="supplied material"):
        AxisymmetricMaterialProblemContract(
            problem_id="dangling-material",
            materials=(curve,),
            regions=(
                MaterialRegionContract(
                    "air",
                    "missing",
                    ConstitutiveLawKind.LINEAR_ISOTROPIC,
                    bounds,
                ),
            ),
            interfaces=(),
            magnetization_sources=(),
            open_boundary_policy=OpenBoundaryDomainPolicy(),
        )
    with pytest.raises(MagneticsValidationError, match="different regions"):
        MaterialInterfaceContract("bad", "same", "same", VectorRZ(1.0, 0.0))
    with pytest.raises(MagneticsValidationError, match="typed magnetics"):
        AxisymmetricMaterialProblemContract(  # type: ignore[arg-type]
            problem_id="untyped-material",
            materials=({"material_id": "air"},),
            regions=(),
            interfaces=(),
            magnetization_sources=(),
            open_boundary_policy=OpenBoundaryDomainPolicy(),
        )


def test_recoil_and_equivalent_current_authorities_are_mutually_exclusive() -> None:
    permanent = checked_synthetic_smco_like_magnet()
    bounds = AxisymmetricBounds(0.01, 0.02, -0.01, 0.01)
    recoil_region = MaterialRegionContract(
        "recoil-region",
        permanent.material_id,
        ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL,
        bounds,
        permanent_magnet_representation=(
            PermanentMagnetRepresentation.RECOIL_REMANENCE
        ),
        magnetization_direction_rz=VectorRZ(0.0, 1.0),
    )
    recoil_contract = AxisymmetricMaterialProblemContract(
        problem_id="recoil-only",
        materials=(permanent,),
        regions=(recoil_region,),
        interfaces=(),
        magnetization_sources=(),
        open_boundary_policy=OpenBoundaryDomainPolicy(),
    )
    assert recoil_contract.regions[0].permanent_magnet_representation is (
        PermanentMagnetRepresentation.RECOIL_REMANENCE
    )
    with pytest.raises(MagneticsValidationError, match="radial magnetization"):
        MaterialRegionContract(
            "axis-recoil",
            permanent.material_id,
            ConstitutiveLawKind.PERMANENT_MAGNET_RECOIL,
            AxisymmetricBounds(0.0, 0.02, -0.01, 0.01),
            permanent_magnet_representation=(
                PermanentMagnetRepresentation.RECOIL_REMANENCE
            ),
            magnetization_direction_rz=VectorRZ(1.0, 0.0),
        )

    host = LinearPermeability(
        "recoil-host", permanent.recoil_relative_permeability
    )
    host_region = MaterialRegionContract(
        "host-region",
        host.material_id,
        ConstitutiveLawKind.LINEAR_ISOTROPIC,
        bounds,
    )
    source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="equivalent-source",
        region_id=host_region.region_id,
        material=permanent,
        bounds=bounds,
        direction=VectorRZ(0.0, 1.0),
        temperature_k=permanent.reference_temperature_k,
    )
    with pytest.raises(MagneticsValidationError, match="cannot use"):
        AxisymmetricMaterialProblemContract(
            problem_id="double-counted",
            materials=(permanent, host),
            regions=(recoil_region, host_region),
            interfaces=(),
            magnetization_sources=(source,),
            open_boundary_policy=OpenBoundaryDomainPolicy(),
        )


def test_handoff_rejects_kind_mismatch_recoil_mismatch_and_duplicate_ids() -> None:
    permanent = checked_synthetic_smco_like_magnet()
    bounds = AxisymmetricBounds(0.01, 0.02, -0.01, 0.01)
    wrong_host = LinearPermeability("wrong-host", 1.0)
    host_region = MaterialRegionContract(
        "host",
        wrong_host.material_id,
        ConstitutiveLawKind.LINEAR_ISOTROPIC,
        bounds,
    )
    source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="source",
        region_id=host_region.region_id,
        material=permanent,
        bounds=bounds,
        direction=VectorRZ(0.0, 1.0),
        temperature_k=permanent.reference_temperature_k,
    )
    with pytest.raises(MagneticsValidationError, match="must match"):
        AxisymmetricMaterialProblemContract(
            "wrong-recoil",
            (permanent, wrong_host),
            (host_region,),
            (),
            (source,),
            OpenBoundaryDomainPolicy(),
        )

    with pytest.raises(MagneticsValidationError, match="incompatible"):
        AxisymmetricMaterialProblemContract(
            "wrong-kind",
            (wrong_host,),
            (
                MaterialRegionContract(
                    "wrong-kind-region",
                    wrong_host.material_id,
                    ConstitutiveLawKind.TABULATED_SINGLE_VALUED,
                    bounds,
                ),
            ),
            (),
            (),
            OpenBoundaryDomainPolicy(),
        )

    interface = MaterialInterfaceContract(
        "duplicate-interface", "a", "b", VectorRZ(1.0, 0.0)
    )
    region_a = MaterialRegionContract(
        "a", wrong_host.material_id, ConstitutiveLawKind.LINEAR_ISOTROPIC, bounds
    )
    region_b = MaterialRegionContract(
        "b", wrong_host.material_id, ConstitutiveLawKind.LINEAR_ISOTROPIC, bounds
    )
    with pytest.raises(MagneticsValidationError, match="interface identifiers"):
        AxisymmetricMaterialProblemContract(
            "duplicates",
            (wrong_host,),
            (region_a, region_b),
            (interface, interface),
            (),
            OpenBoundaryDomainPolicy(),
        )

    matching_host = LinearPermeability(
        "matching-host", permanent.recoil_relative_permeability
    )
    matching_region = MaterialRegionContract(
        "matching-region",
        matching_host.material_id,
        ConstitutiveLawKind.LINEAR_ISOTROPIC,
        bounds,
    )
    duplicate_source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="duplicate-source",
        region_id=matching_region.region_id,
        material=permanent,
        bounds=bounds,
        direction=VectorRZ(0.0, 1.0),
        temperature_k=permanent.reference_temperature_k,
    )
    with pytest.raises(MagneticsValidationError, match="source identifiers"):
        AxisymmetricMaterialProblemContract(
            "duplicate-sources",
            (permanent, matching_host),
            (matching_region,),
            (),
            (duplicate_source, duplicate_source),
            OpenBoundaryDomainPolicy(),
        )
    with pytest.raises(MagneticsValidationError, match="material identifiers"):
        AxisymmetricMaterialProblemContract(
            "duplicate-materials",
            (matching_host, matching_host),
            (),
            (),
            (),
            OpenBoundaryDomainPolicy(),
        )
    with pytest.raises(MagneticsValidationError, match="region identifiers"):
        AxisymmetricMaterialProblemContract(
            "duplicate-regions",
            (matching_host,),
            (matching_region, matching_region),
            (),
            (),
            OpenBoundaryDomainPolicy(),
        )


def test_handoff_rejects_source_material_parameter_mismatch() -> None:
    declared = checked_synthetic_smco_like_magnet()
    inconsistent = SmCoPermanentMagnet(
        material_id=declared.material_id,
        remanence_ref_t=declared.remanence_ref_t * 0.5,
        intrinsic_coercivity_ref_a_per_m=declared.intrinsic_coercivity_ref_a_per_m,
        recoil_relative_permeability=declared.recoil_relative_permeability,
        reference_temperature_k=declared.reference_temperature_k,
        remanence_temp_coefficient_per_k=declared.remanence_temp_coefficient_per_k,
        coercivity_temp_coefficient_per_k=declared.coercivity_temp_coefficient_per_k,
        valid_temperature_min_k=declared.valid_temperature_min_k,
        valid_temperature_max_k=declared.valid_temperature_max_k,
        provenance="deliberately inconsistent test material",
        is_synthetic=True,
    )
    host = LinearPermeability("host", declared.recoil_relative_permeability)
    bounds = AxisymmetricBounds(0.01, 0.02, -0.01, 0.01)
    region = MaterialRegionContract(
        "host-region", host.material_id, ConstitutiveLawKind.LINEAR_ISOTROPIC, bounds
    )
    source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="inconsistent-source",
        region_id=region.region_id,
        material=inconsistent,
        bounds=bounds,
        direction=VectorRZ(0.0, 1.0),
        temperature_k=inconsistent.reference_temperature_k,
    )
    with pytest.raises(MagneticsValidationError, match="parameters do not match"):
        AxisymmetricMaterialProblemContract(
            "inconsistent-source-material",
            (declared, host),
            (region,),
            (),
            (source,),
            OpenBoundaryDomainPolicy(),
        )
