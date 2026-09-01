"""Strict canonical persistence for axisymmetric magnetics hand-off contracts."""

from __future__ import annotations

import json
from hmac import compare_digest

from .common import (
    MagneticsValidationError,
    VectorRZ,
    canonical_json,
    content_sha256,
)
from .contracts import (
    AxisymmetricMaterialProblemContract,
    ConstitutiveLawKind,
    MaterialInterfaceContract,
    MaterialRegionContract,
    OpenBoundaryDomainPolicy,
)
from .materials import (
    ExtrapolationPolicy,
    LinearPermeability,
    SmCoPermanentMagnet,
    TabulatedBHCurve,
)
from .sources import (
    AxisymmetricBounds,
    PermanentMagnetRepresentation,
    UniformAxisymmetricMagnetizationSource,
)

HANDOFF_SCHEMA = "cft_revival.magnetics.axisymmetric_handoff/1.0.0"


def _reject_constant(value: str) -> object:
    raise MagneticsValidationError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MagneticsValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MagneticsValidationError(f"{name} must be a JSON object")
    return value


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise MagneticsValidationError(f"{name} must be a JSON array")
    return value


def _exact_keys(mapping: dict[str, object], expected: set[str], name: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise MagneticsValidationError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _vector(value: object, name: str) -> VectorRZ:
    mapping = _mapping(value, name)
    _exact_keys(mapping, {"radial", "axial"}, name)
    return VectorRZ(float(mapping["radial"]), float(mapping["axial"]))


def _bounds(value: object, name: str) -> AxisymmetricBounds:
    mapping = _mapping(value, name)
    _exact_keys(
        mapping,
        {"r_inner_m", "r_outer_m", "z_min_m", "z_max_m"},
        name,
    )
    return AxisymmetricBounds(
        float(mapping["r_inner_m"]),
        float(mapping["r_outer_m"]),
        float(mapping["z_min_m"]),
        float(mapping["z_max_m"]),
    )


def _material(value: object) -> LinearPermeability | TabulatedBHCurve | SmCoPermanentMagnet:
    mapping = _mapping(value, "material")
    kind = mapping.get("kind")
    if kind == "linear_isotropic":
        _exact_keys(
            mapping,
            {
                "kind",
                "material_id",
                "relative_permeability",
                "permeability_h_per_m",
            },
            "linear material",
        )
        return LinearPermeability(
            str(mapping["material_id"]), float(mapping["relative_permeability"])
        )
    if kind == "tabulated_odd_symmetric_single_valued_b_h":
        _exact_keys(
            mapping,
            {
                "kind",
                "material_id",
                "h_a_per_m",
                "b_t",
                "interpolation",
                "extrapolation",
                "provenance",
                "is_synthetic",
                "hysteresis",
            },
            "tabulated material",
        )
        h_values = tuple(
            float(value) for value in _sequence(mapping["h_a_per_m"], "h_a_per_m")
        )
        b_values = tuple(
            float(value) for value in _sequence(mapping["b_t"], "b_t")
        )
        return TabulatedBHCurve(
            material_id=str(mapping["material_id"]),
            h_a_per_m=h_values,
            b_t=b_values,
            extrapolation=ExtrapolationPolicy(str(mapping["extrapolation"])),
            provenance=str(mapping["provenance"]),
            is_synthetic=bool(mapping["is_synthetic"]),
        )
    if kind == "smco_like_linear_recoil_permanent_magnet":
        expected = {
            "kind",
            "material_id",
            "remanence_ref_t",
            "intrinsic_coercivity_ref_a_per_m",
            "recoil_relative_permeability",
            "reference_temperature_k",
            "remanence_temp_coefficient_per_k",
            "coercivity_temp_coefficient_per_k",
            "valid_temperature_min_k",
            "valid_temperature_max_k",
            "provenance",
            "is_synthetic",
        }
        _exact_keys(mapping, expected, "permanent-magnet material")
        return SmCoPermanentMagnet(
            material_id=str(mapping["material_id"]),
            remanence_ref_t=float(mapping["remanence_ref_t"]),
            intrinsic_coercivity_ref_a_per_m=float(
                mapping["intrinsic_coercivity_ref_a_per_m"]
            ),
            recoil_relative_permeability=float(
                mapping["recoil_relative_permeability"]
            ),
            reference_temperature_k=float(mapping["reference_temperature_k"]),
            remanence_temp_coefficient_per_k=float(
                mapping["remanence_temp_coefficient_per_k"]
            ),
            coercivity_temp_coefficient_per_k=float(
                mapping["coercivity_temp_coefficient_per_k"]
            ),
            valid_temperature_min_k=float(mapping["valid_temperature_min_k"]),
            valid_temperature_max_k=float(mapping["valid_temperature_max_k"]),
            provenance=str(mapping["provenance"]),
            is_synthetic=bool(mapping["is_synthetic"]),
        )
    raise MagneticsValidationError(f"unknown material kind {kind!r}")


def _region(value: object) -> MaterialRegionContract:
    mapping = _mapping(value, "region")
    _exact_keys(
        mapping,
        {
            "region_id",
            "constitutive_law_id",
            "constitutive_law_kind",
            "bounds",
            "priority",
            "permanent_magnet_representation",
            "magnetization_direction_rz",
        },
        "region",
    )
    representation_value = mapping["permanent_magnet_representation"]
    direction_value = mapping["magnetization_direction_rz"]
    return MaterialRegionContract(
        region_id=str(mapping["region_id"]),
        constitutive_law_id=str(mapping["constitutive_law_id"]),
        constitutive_law_kind=ConstitutiveLawKind(
            str(mapping["constitutive_law_kind"])
        ),
        bounds=_bounds(mapping["bounds"], "region bounds"),
        priority=int(mapping["priority"]),
        permanent_magnet_representation=(
            None
            if representation_value is None
            else PermanentMagnetRepresentation(str(representation_value))
        ),
        magnetization_direction_rz=(
            None
            if direction_value is None
            else _vector(direction_value, "magnetization direction")
        ),
    )


def _interface(value: object) -> MaterialInterfaceContract:
    mapping = _mapping(value, "interface")
    _exact_keys(
        mapping,
        {
            "interface_id",
            "minus_region_id",
            "plus_region_id",
            "normal_minus_to_plus_rz",
            "free_surface_current_phi_a_per_m",
            "required_jump_conditions",
        },
        "interface",
    )
    return MaterialInterfaceContract(
        interface_id=str(mapping["interface_id"]),
        minus_region_id=str(mapping["minus_region_id"]),
        plus_region_id=str(mapping["plus_region_id"]),
        normal_minus_to_plus_rz=_vector(
            mapping["normal_minus_to_plus_rz"], "interface normal"
        ),
        free_surface_current_phi_a_per_m=float(
            mapping["free_surface_current_phi_a_per_m"]
        ),
    )


def _source(
    value: object,
    materials_by_id: dict[
        str, LinearPermeability | TabulatedBHCurve | SmCoPermanentMagnet
    ],
) -> UniformAxisymmetricMagnetizationSource:
    mapping = _mapping(value, "source")
    _exact_keys(
        mapping,
        {
            "kind",
            "source_id",
            "permanent_magnet_material_id",
            "region_id",
            "representation",
            "bounds",
            "magnetization_direction_rz",
            "magnetization_a_per_m",
            "temperature_k",
            "bound_volume_current_density_phi_a_per_m2",
            "equivalent_bound_current_sheets",
        },
        "source",
    )
    temperature = mapping["temperature_k"]
    material_id = str(mapping["permanent_magnet_material_id"])
    material = materials_by_id.get(material_id)
    if not isinstance(material, SmCoPermanentMagnet):
        raise MagneticsValidationError(
            "source permanent_magnet_material_id must reference a supplied "
            "permanent-magnet material"
        )
    if temperature is None:
        raise MagneticsValidationError("source temperature_k must not be null")
    return UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id=str(mapping["source_id"]),
        region_id=str(mapping["region_id"]),
        material=material,
        bounds=_bounds(mapping["bounds"], "source bounds"),
        direction=_vector(
            mapping["magnetization_direction_rz"], "source magnetization direction"
        ),
        temperature_k=float(temperature),
    )


def _policy(value: object) -> OpenBoundaryDomainPolicy:
    mapping = _mapping(value, "open boundary policy")
    _exact_keys(
        mapping,
        {
            "kind",
            "minimum_padding_characteristic_lengths",
            "maximum_boundary_to_peak_field_ratio",
            "domain_expansion_factor",
            "required_expansion_comparisons",
            "maximum_qoi_relative_change",
            "claim_limit",
        },
        "open boundary policy",
    )
    return OpenBoundaryDomainPolicy(
        minimum_padding_characteristic_lengths=float(
            mapping["minimum_padding_characteristic_lengths"]
        ),
        maximum_boundary_to_peak_field_ratio=float(
            mapping["maximum_boundary_to_peak_field_ratio"]
        ),
        domain_expansion_factor=float(mapping["domain_expansion_factor"]),
        required_expansion_comparisons=int(
            mapping["required_expansion_comparisons"]
        ),
        maximum_qoi_relative_change=float(
            mapping["maximum_qoi_relative_change"]
        ),
    )


def serialize_handoff(contract: AxisymmetricMaterialProblemContract) -> str:
    """Serialize canonical content with a SHA-256 tamper-evident envelope."""

    content = contract.to_dict()
    return canonical_json(
        {
            "schema": HANDOFF_SCHEMA,
            "content_sha256": content_sha256(content),
            "content": content,
        }
    )


def deserialize_handoff(serialized: str) -> AxisymmetricMaterialProblemContract:
    """Strictly verify and reconstruct a closed hand-off contract."""

    if not isinstance(serialized, str):
        raise MagneticsValidationError("serialized handoff must be text")
    try:
        decoded = json.loads(
            serialized,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError) as error:
        if isinstance(error, MagneticsValidationError):
            raise
        raise MagneticsValidationError("invalid handoff JSON") from error
    envelope = _mapping(decoded, "handoff envelope")
    _exact_keys(
        envelope,
        {"schema", "content_sha256", "content"},
        "handoff envelope",
    )
    if canonical_json(envelope) != serialized:
        raise MagneticsValidationError("handoff JSON is not in canonical form")
    if envelope["schema"] != HANDOFF_SCHEMA:
        raise MagneticsValidationError("unsupported handoff schema")
    content = _mapping(envelope["content"], "handoff content")
    digest = str(envelope["content_sha256"])
    if len(digest) != 64 or not compare_digest(digest, content_sha256(content)):
        raise MagneticsValidationError("handoff SHA-256 digest verification failed")
    _exact_keys(
        content,
        {
            "contract_version",
            "coordinate_system",
            "problem_id",
            "materials",
            "regions",
            "interfaces",
            "magnetization_sources",
            "open_boundary_policy",
            "solver_requirements",
        },
        "handoff content",
    )
    try:
        materials = tuple(
            _material(value)
            for value in _sequence(content["materials"], "materials")
        )
        materials_by_id = {material.material_id: material for material in materials}
        contract = AxisymmetricMaterialProblemContract(
            problem_id=str(content["problem_id"]),
            materials=materials,
            regions=tuple(
                _region(value)
                for value in _sequence(content["regions"], "regions")
            ),
            interfaces=tuple(
                _interface(value)
                for value in _sequence(content["interfaces"], "interfaces")
            ),
            magnetization_sources=tuple(
                _source(value, materials_by_id)
                for value in _sequence(
                    content["magnetization_sources"], "magnetization_sources"
                )
            ),
            open_boundary_policy=_policy(content["open_boundary_policy"]),
            contract_version=str(content["contract_version"]),
        )
    except MagneticsValidationError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise MagneticsValidationError(
            "handoff contains an invalid closed-schema value"
        ) from error
    if canonical_json(contract.to_dict()) != canonical_json(content):
        raise MagneticsValidationError(
            "handoff content contains altered derived or discriminator fields"
        )
    return contract
