import json
import subprocess
import sys

import pytest

from cft_revival.magnetics import (
    AxisymmetricBounds,
    AxisymmetricMaterialProblemContract,
    ConstitutiveLawKind,
    LinearPermeability,
    MagneticsValidationError,
    MaterialRegionContract,
    OpenBoundaryDomainPolicy,
    UniformAxisymmetricMagnetizationSource,
    VectorRZ,
    canonical_json,
    checked_synthetic_smco_like_magnet,
    content_sha256,
    deserialize_handoff,
    serialize_handoff,
)


def _handoff() -> AxisymmetricMaterialProblemContract:
    permanent = checked_synthetic_smco_like_magnet()
    host = LinearPermeability(
        "permanent-magnet-recoil-host", permanent.recoil_relative_permeability
    )
    bounds = AxisymmetricBounds(0.01, 0.02, -0.02, 0.02)
    region = MaterialRegionContract(
        "magnet-region",
        host.material_id,
        ConstitutiveLawKind.LINEAR_ISOTROPIC,
        bounds,
    )
    source = UniformAxisymmetricMagnetizationSource.from_permanent_magnet(
        source_id="magnet-source",
        region_id=region.region_id,
        material=permanent,
        bounds=bounds,
        direction=VectorRZ(0.0, 1.0),
        temperature_k=permanent.reference_temperature_k,
    )
    return AxisymmetricMaterialProblemContract(
        problem_id="digest-round-trip",
        materials=(permanent, host),
        regions=(region,),
        interfaces=(),
        magnetization_sources=(source,),
        open_boundary_policy=OpenBoundaryDomainPolicy(),
    )


def test_signed_zero_is_canonical_and_digest_is_cross_process() -> None:
    value = {"positive": 0.0, "negative": -0.0, "nested": [-0.0]}
    assert canonical_json(value) == '{"negative":0.0,"nested":[0.0],"positive":0.0}'
    local_digest = content_sha256(value)
    script = (
        "import sys;"
        "sys.path.insert(0,'src');"
        "from cft_revival.magnetics import content_sha256;"
        "print(content_sha256({'positive':0.0,'negative':-0.0,'nested':[-0.0]}))"
    )
    remote_digest = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=".",
        text=True,
    ).strip()
    assert remote_digest == local_digest
    assert len(local_digest) == 64
    with pytest.raises(MagneticsValidationError, match="keys must be strings"):
        canonical_json({1: "ambiguous"})  # type: ignore[dict-item]


def test_strict_handoff_round_trip_and_digest_verification() -> None:
    contract = _handoff()
    serialized = serialize_handoff(contract)
    restored = deserialize_handoff(serialized)
    assert restored == contract
    envelope = json.loads(serialized)
    assert envelope["content_sha256"] == content_sha256(envelope["content"])


def test_handoff_rejects_tampering_noncanonical_and_unknown_fields() -> None:
    serialized = serialize_handoff(_handoff())
    envelope = json.loads(serialized)
    envelope["content"]["problem_id"] = "tampered"
    with pytest.raises(MagneticsValidationError, match="digest"):
        deserialize_handoff(canonical_json(envelope))

    envelope = json.loads(serialized)
    envelope["content"]["materials"][1]["permeability_h_per_m"] = 999.0
    envelope["content_sha256"] = content_sha256(envelope["content"])
    with pytest.raises(MagneticsValidationError, match="altered derived"):
        deserialize_handoff(canonical_json(envelope))

    with pytest.raises(MagneticsValidationError, match="canonical"):
        deserialize_handoff(serialized + "\n")

    envelope = json.loads(serialized)
    envelope["content"]["unknown"] = True
    envelope["content_sha256"] = content_sha256(envelope["content"])
    with pytest.raises(MagneticsValidationError, match="extra"):
        deserialize_handoff(canonical_json(envelope))


def test_handoff_rejects_duplicate_keys_and_unknown_discriminators() -> None:
    with pytest.raises(MagneticsValidationError, match="duplicate JSON key"):
        deserialize_handoff(
            '{"content":{},"content_sha256":"x","schema":"x","schema":"y"}'
        )

    envelope = json.loads(serialize_handoff(_handoff()))
    envelope["content"]["materials"][0]["kind"] = "unknown-material"
    envelope["content_sha256"] = content_sha256(envelope["content"])
    with pytest.raises(MagneticsValidationError, match="unknown material"):
        deserialize_handoff(canonical_json(envelope))


def test_rehashed_inconsistent_source_magnitude_and_temperature_are_rejected() -> None:
    contract = _handoff()
    source = contract.magnetization_sources[0]
    assert source.magnetization_a_per_m.magnitude == pytest.approx(835_563.451)

    envelope = json.loads(serialize_handoff(contract))
    source_content = envelope["content"]["magnetization_sources"][0]
    source_content["magnetization_a_per_m"] = {"radial": 0.0, "axial": 1.0}
    envelope["content_sha256"] = content_sha256(envelope["content"])
    with pytest.raises(MagneticsValidationError, match="altered derived"):
        deserialize_handoff(canonical_json(envelope))

    envelope = json.loads(serialize_handoff(contract))
    envelope["content"]["magnetization_sources"][0]["temperature_k"] = 1.0
    envelope["content_sha256"] = content_sha256(envelope["content"])
    with pytest.raises(MagneticsValidationError, match="outside"):
        deserialize_handoff(canonical_json(envelope))
