from hashlib import sha256
import json

import pytest

import cft_revival.hybrid.warp_backend as warp_backend
from cft_revival.hybrid import (
    ELEMENTARY_CHARGE_C,
    CartesianGrid1D,
    HybridCheckpoint,
    HybridDeviceError,
    HybridOptionalDependencyError,
    HybridValidationError,
    Particle,
    ProvenanceRecord,
    UniformFields,
    XE,
    XE_DOUBLE_PLUS,
    XE_PLUS,
    XenonSpecies,
    checkpoint_digest,
    deposit_cic_periodic,
    load_checkpoint,
    save_checkpoint,
)
from cft_revival.hybrid.reference import boris_push
from cft_revival.hybrid.warp_backend import (
    deposit_cic_periodic_warp,
    device_available,
    push_boris_warp,
)


def _particles() -> tuple[Particle, ...]:
    return (
        Particle(0, XE, (0.05, 0.0, 0.0), (3.0, 1.0, -1.0), weight=2.0),
        Particle(1, XE_PLUS, (0.31, 0.0, 0.0), (5.0, -2.0, 4.0)),
        Particle(2, XE_DOUBLE_PLUS, (0.79, 0.0, 0.0), (-1.0, 6.0, 2.0), weight=3.0),
    )


def _rehash_and_write(path, envelope) -> None:
    canonical = json.dumps(
        envelope["payload"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    envelope["sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    path.write_text(json.dumps(envelope), encoding="utf-8")


def test_checkpoint_round_trip_digest_and_tamper_detection(tmp_path) -> None:
    checkpoint = HybridCheckpoint(
        step=7,
        time_s=7.0e-8,
        dt_s=1.0e-8,
        rng_seed=123456,
        particles=_particles(),
        provenance=ProvenanceRecord(
            model_scope="prescribed-field manufactured verification",
            created_utc="2026-09-01T13:00:00Z",
            code_revision=None,
            cross_section_provenance="synthetic constant verification fixture",
            notes=("not a calibrated thruster prediction",),
        ),
    )
    path = tmp_path / "hybrid-checkpoint.json"
    save_checkpoint(checkpoint, path)
    restored = load_checkpoint(path)
    assert restored == checkpoint
    assert checkpoint_digest(restored) == checkpoint_digest(checkpoint)

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["step"] = 8
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(HybridValidationError, match="SHA-256"):
        load_checkpoint(path)


def test_checkpoint_preserves_complete_custom_species(tmp_path) -> None:
    custom = XenonSpecies(
        "Xe+",
        1,
        mass_kg=2.25e-25,
        identifier="xe-plus-custom-fixture",
        charge_c_override=ELEMENTARY_CHARGE_C,
    )
    particle = Particle(
        444,
        custom,
        (0.1, 0.2, 0.3),
        (4.0, 5.0, 6.0),
    )
    checkpoint = HybridCheckpoint(
        step=1,
        time_s=1.0e-8,
        dt_s=1.0e-8,
        rng_seed=99,
        particles=(particle,),
        provenance=ProvenanceRecord(
            model_scope="custom species checkpoint fixture",
            created_utc="2026-09-01T13:00:00Z",
            code_revision=None,
            cross_section_provenance="synthetic",
        ),
    )
    path = tmp_path / "custom.json"
    save_checkpoint(checkpoint, path)
    restored = load_checkpoint(path)
    assert restored.particles[0].species == custom
    assert restored.particles[0].species.identifier == custom.identifier
    assert restored.particles[0].species.mass_kg == custom.mass_kg
    assert restored.particles[0].species.charge_c == custom.charge_c


@pytest.mark.parametrize(
    ("target", "key", "value", "message"),
    [
        ("payload", "extra", 1, "checkpoint payload"),
        ("particle", "extra", 1, "particle"),
        ("species", "extra", 1, "species"),
        ("rng", "extra", 1, "rng"),
        ("time_levels", "extra", 1, "time_levels"),
        ("provenance", "extra", 1, "provenance"),
    ],
)
def test_checkpoint_rejects_rehashed_extra_fields(
    tmp_path, target: str, key: str, value, message: str
) -> None:
    checkpoint = HybridCheckpoint(
        step=0,
        time_s=0.0,
        dt_s=1.0e-8,
        rng_seed=1,
        particles=_particles(),
        provenance=ProvenanceRecord(
            model_scope="closed schema fixture",
            created_utc="2026-09-01T13:00:00Z",
            code_revision=None,
            cross_section_provenance="synthetic",
        ),
    )
    path = tmp_path / f"{target}.json"
    save_checkpoint(checkpoint, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    targets = {
        "payload": envelope["payload"],
        "particle": envelope["payload"]["particles"][0],
        "species": envelope["payload"]["particles"][0]["species"],
        "rng": envelope["payload"]["rng"],
        "time_levels": envelope["payload"]["time_levels"],
        "provenance": envelope["payload"]["provenance"],
    }
    targets[target][key] = value
    _rehash_and_write(path, envelope)
    with pytest.raises(HybridValidationError, match=message):
        load_checkpoint(path)


def test_checkpoint_rejects_invalid_typed_fields_and_staggering(tmp_path) -> None:
    checkpoint = HybridCheckpoint(
        step=0,
        time_s=0.0,
        dt_s=1.0e-8,
        rng_seed=1,
        particles=_particles(),
        provenance=ProvenanceRecord(
            model_scope="typed schema fixture",
            created_utc="2026-09-01T13:00:00Z",
            code_revision=None,
            cross_section_provenance="synthetic",
        ),
    )
    path = tmp_path / "invalid.json"
    save_checkpoint(checkpoint, path)
    original = json.loads(path.read_text(encoding="utf-8"))

    invalid_alive = json.loads(json.dumps(original))
    invalid_alive["payload"]["particles"][0]["alive"] = 1
    _rehash_and_write(path, invalid_alive)
    with pytest.raises(HybridValidationError, match="boolean"):
        load_checkpoint(path)

    invalid_stagger = json.loads(json.dumps(original))
    invalid_stagger["payload"]["time_levels"]["velocity"] = "n"
    _rehash_and_write(path, invalid_stagger)
    with pytest.raises(HybridValidationError, match="time_levels"):
        load_checkpoint(path)

    invalid_particle_stagger = json.loads(json.dumps(original))
    invalid_particle_stagger["payload"]["particles"][0][
        "velocity_time_level"
    ] = "synchronous_n"
    _rehash_and_write(path, invalid_particle_stagger)
    with pytest.raises(HybridValidationError, match="time levels"):
        load_checkpoint(path)

    nonfinite = json.loads(json.dumps(original))
    nonfinite["payload"]["time_s"] = float("nan")
    nonfinite["sha256"] = "0" * 64
    path.write_text(json.dumps(nonfinite, allow_nan=True), encoding="utf-8")
    with pytest.raises(HybridValidationError, match="finite"):
        load_checkpoint(path)

    invalid_timestamp = json.loads(json.dumps(original))
    invalid_timestamp["payload"]["provenance"]["created_utc"] = "not-a-time"
    _rehash_and_write(path, invalid_timestamp)
    with pytest.raises(HybridValidationError, match="ISO-8601"):
        load_checkpoint(path)

    invalid_notes = json.loads(json.dumps(original))
    invalid_notes["payload"]["provenance"]["notes"] = ["valid", 3]
    _rehash_and_write(path, invalid_notes)
    with pytest.raises(HybridValidationError, match="notes"):
        load_checkpoint(path)

    boolean_time = json.loads(json.dumps(original))
    boolean_time["payload"]["time_s"] = True
    _rehash_and_write(path, boolean_time)
    with pytest.raises(HybridValidationError, match="actual JSON finite"):
        load_checkpoint(path)

    extra_envelope = json.loads(json.dumps(original))
    extra_envelope["extra"] = 1
    path.write_text(json.dumps(extra_envelope), encoding="utf-8")
    with pytest.raises(HybridValidationError, match="valid v1 envelope"):
        load_checkpoint(path)


def test_checkpoint_numbers_are_not_coerced_from_strings(tmp_path) -> None:
    checkpoint = HybridCheckpoint(
        step=0,
        time_s=0.0,
        dt_s=1.0e-8,
        rng_seed=1,
        particles=_particles(),
        provenance=ProvenanceRecord(
            model_scope="strict numeric fixture",
            created_utc="2026-09-01T13:00:00Z",
            code_revision=None,
            cross_section_provenance="synthetic",
        ),
    )
    path = tmp_path / "strict-numeric.json"
    save_checkpoint(checkpoint, path)
    original = json.loads(path.read_text(encoding="utf-8"))
    mutations = (
        lambda payload: payload.__setitem__("time_s", "0.0"),
        lambda payload: payload.__setitem__("dt_s", "1e-8"),
        lambda payload: payload["particles"][0].__setitem__(
            "weight", "2.0"
        ),
        lambda payload: payload["particles"][0]["position_m"].__setitem__(
            0, "0.05"
        ),
        lambda payload: payload["particles"][0][
            "velocity_m_per_s"
        ].__setitem__(0, "3.0"),
        lambda payload: payload["particles"][0]["species"].__setitem__(
            "mass_kg", "2.18e-25"
        ),
        lambda payload: payload["particles"][0]["species"].__setitem__(
            "charge_c", "0.0"
        ),
    )
    for mutate in mutations:
        envelope = json.loads(json.dumps(original))
        mutate(envelope["payload"])
        _rehash_and_write(path, envelope)
        with pytest.raises(
            HybridValidationError, match="actual JSON finite number"
        ):
            load_checkpoint(path)


def test_checkpoint_rejects_charge_invariant_violation(tmp_path) -> None:
    checkpoint = HybridCheckpoint(
        step=0,
        time_s=0.0,
        dt_s=1.0e-8,
        rng_seed=1,
        particles=_particles(),
        provenance=ProvenanceRecord(
            model_scope="charge invariant fixture",
            created_utc="2026-09-01T13:00:00Z",
            code_revision=None,
            cross_section_provenance="synthetic",
        ),
    )
    path = tmp_path / "bad-charge.json"
    save_checkpoint(checkpoint, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["particles"][1]["species"]["charge_c"] = 0.0
    _rehash_and_write(path, envelope)
    with pytest.raises(HybridValidationError, match="elementary charge"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"sha256":"0","sha256":"1","payload":{}}',
        (
            '{"sha256":"'
            + "0" * 64
            + '","payload":{"rng":{"seed":1,"seed":2}}}'
        ),
    ],
)
def test_checkpoint_rejects_top_level_and_nested_duplicate_keys(
    tmp_path, raw_json: str
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(raw_json, encoding="utf-8")
    with pytest.raises(HybridValidationError, match="duplicate JSON object key"):
        load_checkpoint(path)


def test_checkpoint_integrity_is_not_authenticity(tmp_path) -> None:
    checkpoint = HybridCheckpoint(
        step=0,
        time_s=0.0,
        dt_s=1.0e-8,
        rng_seed=1,
        particles=_particles(),
        provenance=ProvenanceRecord(
            model_scope="integrity fixture",
            created_utc="2026-09-01T13:00:00Z",
            code_revision=None,
            cross_section_provenance="synthetic",
        ),
    )
    path = tmp_path / "rehashed.json"
    save_checkpoint(checkpoint, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["step"] = 12
    _rehash_and_write(path, envelope)
    assert load_checkpoint(path).step == 12


def test_checkpoint_rejects_bad_provenance_and_duplicate_ids() -> None:
    with pytest.raises(HybridValidationError, match="ISO-8601"):
        ProvenanceRecord(
            model_scope="bad timestamp",
            created_utc="yesterday",
            code_revision=None,
            cross_section_provenance="synthetic",
        )
    with pytest.raises(HybridValidationError, match="tuple"):
        ProvenanceRecord(
            model_scope="bad notes",
            created_utc="2026-09-01T13:00:00Z",
            code_revision=None,
            cross_section_provenance="synthetic",
            notes=["mutable"],  # type: ignore[arg-type]
        )
    duplicate = (_particles()[0], _particles()[0])
    with pytest.raises(HybridValidationError, match="unique"):
        HybridCheckpoint(
            step=0,
            time_s=0.0,
            dt_s=1.0e-8,
            rng_seed=1,
            particles=duplicate,
            provenance=ProvenanceRecord(
                model_scope="duplicate ids",
                created_utc="2026-09-01T13:00:00Z",
                code_revision=None,
                cross_section_provenance="synthetic",
            ),
        )
    conflicting_species = (
        Particle(
            1,
            XenonSpecies(
                "Xe+",
                1,
                identifier="same-id",
                mass_kg=2.2e-25,
            ),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        Particle(
            2,
            XenonSpecies(
                "Xe+",
                1,
                identifier="same-id",
                mass_kg=2.3e-25,
            ),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )
    with pytest.raises(HybridValidationError, match="different properties"):
        HybridCheckpoint(
            step=0,
            time_s=0.0,
            dt_s=1.0e-8,
            rng_seed=1,
            particles=conflicting_species,
            provenance=ProvenanceRecord(
                model_scope="species identity conflict",
                created_utc="2026-09-01T13:00:00Z",
                code_revision=None,
                cross_section_provenance="synthetic",
            ),
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_pusher_and_deposition_match_reference(device: str) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    particles = _particles()
    fields = UniformFields(
        electric_v_per_m=(20.0, -3.0, 1.0),
        magnetic_t=(0.0, 0.0, 0.02),
    )
    dt = 1.0e-8
    expected_particles = tuple(boris_push(particle, fields, dt) for particle in particles)
    actual_particles = push_boris_warp(particles, fields, dt, device=device)
    for actual, expected in zip(actual_particles, expected_particles, strict=True):
        assert actual.position_m == pytest.approx(expected.position_m, rel=2.0e-14, abs=1.0e-30)
        assert actual.velocity_m_per_s == pytest.approx(
            expected.velocity_m_per_s, rel=2.0e-14, abs=1.0e-30
        )

    grid = CartesianGrid1D(0.0, 1.0, 8, transverse_area_m2=0.5)
    expected_moments = deposit_cic_periodic(actual_particles, grid)
    actual_moments = deposit_cic_periodic_warp(actual_particles, grid, device=device)
    assert actual_moments.number_per_m3 == pytest.approx(expected_moments.number_per_m3)
    assert actual_moments.charge_c_per_m3 == pytest.approx(
        expected_moments.charge_c_per_m3, rel=2.0e-14, abs=1.0e-40
    )
    assert actual_moments.kinetic_energy_j_per_m3 == pytest.approx(
        expected_moments.kinetic_energy_j_per_m3, rel=2.0e-14, abs=1.0e-40
    )
    for actual_row, expected_row in zip(
        actual_moments.current_a_per_m2,
        expected_moments.current_a_per_m2,
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row, rel=2.0e-14, abs=1.0e-40)
    for actual_row, expected_row in zip(
        actual_moments.momentum_kg_per_m2_s,
        expected_moments.momentum_kg_per_m2_s,
        strict=True,
    ):
        assert actual_row == pytest.approx(expected_row, rel=2.0e-14, abs=1.0e-40)


def test_warp_zero_step_is_exact_without_launch() -> None:
    if not device_available("cpu"):
        pytest.skip("Warp CPU device is unavailable")
    particles = _particles()
    assert push_boris_warp(particles, UniformFields(), 0.0, device="cpu") == particles


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_warp_empty_batches_match_cpu_semantics(device: str) -> None:
    if not device_available(device):
        pytest.skip(f"Warp device {device} is unavailable")
    grid = CartesianGrid1D(0.0, 1.0, 4)
    assert push_boris_warp((), UniformFields(), 1.0e-8, device=device) == ()
    assert deposit_cic_periodic_warp((), grid, device=device) == deposit_cic_periodic(
        (), grid
    )


def test_missing_warp_and_invalid_devices_fail_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(warp_backend, "wp", None)
    assert warp_backend.warp_available() is False
    assert warp_backend.device_available("cpu") is False
    with pytest.raises(HybridOptionalDependencyError):
        warp_backend.push_boris_warp(_particles(), UniformFields(), 1.0, device="cpu")


def test_invalid_warp_device_and_batch_failures() -> None:
    if not warp_backend.warp_available():
        pytest.skip("Warp is unavailable")
    with pytest.raises(HybridDeviceError):
        push_boris_warp(_particles(), UniformFields(), 1.0e-9, device="gpu")
    with pytest.raises(HybridValidationError, match="outside"):
        deposit_cic_periodic_warp(
            (Particle(99, XE, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),),
            CartesianGrid1D(0.0, 1.0, 4),
            device="cpu",
        )
