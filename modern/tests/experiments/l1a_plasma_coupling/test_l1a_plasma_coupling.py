from __future__ import annotations

import hashlib
import shutil
from datetime import timedelta
from pathlib import Path

import pytest

from cft_revival.coupling import (
    EvidenceVerificationError,
    MapValidationPolicy,
    build_screening_proxy,
)
from cft_revival.plasma import (
    PlasmaMultiStartResult,
    PlasmaState,
    SolverOptions,
    XenonGlobalInputs,
    solve_global_discharge,
)
from experiments.l1a_plasma_coupling.adapter import (
    ACCEPTANCE_TIME_UTC,
    load_accepted_evidence,
)
from experiments.l1a_plasma_coupling.experiment import (
    UNCERTAINTY,
    build_plasma_inputs,
    run_experiment,
    run_plasma_case,
    serialize_plasma,
    topology_compatibility,
    validate_bundle,
)

MODERN = Path(__file__).resolve().parents[3]
ACCEPTED = MODERN / "examples" / "axisymmetric" / "results"
MANIFEST = ACCEPTED / "manifest-l1a-v1.json"
TRIPLET = ACCEPTED / "hypothetical-thick-outer-triplet-l1a-v1.json"


@pytest.fixture(scope="module")
def experiment_bundle(tmp_path_factory):
    return run_experiment(tmp_path_factory.mktemp("l1a-plasma-results"))


def _copy_with_sidecar(source: Path, target: Path) -> None:
    shutil.copyfile(source, target)
    shutil.copyfile(
        source.with_name(source.name + ".sha256"),
        target.with_name(target.name + ".sha256"),
    )


def test_adapter_rejects_artifact_tampering_and_manifest_mismatch(tmp_path) -> None:
    artifact = tmp_path / TRIPLET.name
    manifest = tmp_path / MANIFEST.name
    _copy_with_sidecar(TRIPLET, artifact)
    _copy_with_sidecar(MANIFEST, manifest)
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="sidecar"):
        load_accepted_evidence(artifact, manifest)

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact.with_name(artifact.name + ".sha256").write_text(
        f"{digest}  {artifact.name}\n", encoding="ascii", newline="\n"
    )
    with pytest.raises(ValueError, match="manifest file hash"):
        load_accepted_evidence(artifact, manifest)


def test_adapter_rejects_stale_evidence() -> None:
    with pytest.raises(EvidenceVerificationError, match="stale"):
        load_accepted_evidence(
            TRIPLET,
            MANIFEST,
            policy=MapValidationPolicy(maximum_age_s=1.0),
            reference_time_utc=ACCEPTANCE_TIME_UTC + timedelta(seconds=2),
        )


def test_real_topology_mismatch_is_not_forced_into_four_cells() -> None:
    opposed = ACCEPTED / "hypothetical-opposed-cusp-l1a-v1.json"
    evidence, _ = load_accepted_evidence(opposed, MANIFEST)
    # Same deprecated coupling v2 same-z projection the experiment uses.
    record = build_screening_proxy(
        evidence,
        wall_radius_m=0.1,
        uncertainty_model=UNCERTAINTY,
        reference_time_utc=ACCEPTANCE_TIME_UTC,
    )
    compatible, reason = topology_compatibility(record)
    assert not compatible
    assert len(record.segments) == 3
    assert "requires exactly four" in reason


def test_hash_identity_is_end_to_end_and_source_bound(experiment_bundle) -> None:
    dataset = experiment_bundle["dataset"]
    triplet = next(item for item in dataset["designs"] if item["design_id"] == "triplet")
    identity = triplet["coupling_identity"]
    assert identity["artifact_hash"] == hashlib.sha256(TRIPLET.read_bytes()).hexdigest()
    for key in (
        "record_hash",
        "field_map_hash",
        "artifact_hash",
        "source_hash",
        "source_map_binding_hash",
        "field_model_hash",
        "code_hash",
        "config_hash",
        "adapter_code_hash",
        "coupling_model_hash",
    ):
        assert len(identity[key]) == 64
    identity_kinds = {
        identity["artifact_hash"],
        identity["field_map_hash"],
        identity["source_hash"],
    }
    assert len(identity_kinds) == 3
    diagnostics = identity["diagnostics"]
    assert diagnostics["residual_norm"] <= diagnostics["residual_tolerance"]


def test_experiment_is_repeatable_and_bundle_is_hash_pinned(
    experiment_bundle, tmp_path
) -> None:
    repeated = run_experiment(tmp_path)
    assert repeated["dataset"] == experiment_bundle["dataset"]
    loaded = validate_bundle(tmp_path)
    assert loaded["dataset"]["integrity"] == repeated["dataset"]["integrity"]


def test_failures_publish_neither_plasma_state_nor_performance(experiment_bundle) -> None:
    dataset = experiment_bundle["dataset"]
    failed = [item for item in dataset["designs"] if item["status"] == "failed"]
    assert {item["design_id"] for item in failed} == {
        "compact",
        "opposed-cusp",
        "triplet",
    }
    assert all(item["plasma"] is None for item in failed)
    assert all(
        item["screening_performance"]["status"] == "not_published"
        for item in failed
    )
    assert dataset["summary"] == {
        "design_count": 3,
        "compatible_count": 0,
        "accepted_count": 0,
        "failed_count": 3,
        "plasma_solve_count": 0,
        "performance_publication_count": 0,
    }


def test_tolerance_failure_retains_rows_without_false_state_publication() -> None:
    inputs = XenonGlobalInputs(1000.0, 1.0, (0.060, 0.119, 0.160, 0.254))
    rounded = PlasmaState(
        plasma_potential_v=(14.1, 1000.0, 1000.0, 1000.0),
        electron_temperature_ev=(8.9, 100.1, 43.1, 23.5),
        ionization_source_current_a=(0.008, 0.543, 0.310, 0.157),
        electron_current_a=(0.106, 0.107, 0.637, 0.845, 1.002),
        ion_current_a=(0.894, 0.893, 0.363, 0.155, -0.002),
        cusp_ion_current_a=(0.007, 0.013, 0.102),
    )
    failed = solve_global_discharge(
        inputs,
        rounded,
        options=SolverOptions(max_iterations=2, residual_tolerance=1.0e-12),
    )
    serialized = serialize_plasma(
        PlasmaMultiStartResult(
            best=failed,
            attempts=(failed,),
            selected_start_index=0,
            residual_floor=failed.diagnostics.residual_inf_norm,
        )
    )
    assert not serialized["valid_state"]
    assert not serialized["attempts"][0]["state_published"]
    assert len(serialized["attempts"][0]["residual_rows"]) == 28
    assert all(row["raw_si"] is None for row in serialized["attempts"][0]["residual_rows"])


def test_successful_manufactured_coupled_fixture_is_repeatable() -> None:
    voltage = 1000.0
    phi = (10.0, 500.0, 800.0, 1000.0)
    electron = [0.002 * phi[0] ** 1.5]
    source = []
    temperature = []
    for cell in range(4):
        gain = phi[cell] - (0.0 if cell == 0 else phi[cell - 1])
        if cell:
            gain += temperature[cell - 1]
        ionization = electron[cell] * 0.07 * gain / 12.1
        source.append(ionization)
        transported = electron[cell] + ionization
        temperature.append(0.68 * electron[cell] * gain / transported)
        if cell < 3:
            electron.append(transported)
    electron.append(electron[3] + source[3])
    inputs = build_plasma_inputs(
        (0.0, 0.0, 0.0, 0.0),
        anode_voltage_v=voltage,
        anode_current_a=electron[4],
    )
    state = PlasmaState(
        plasma_potential_v=phi,
        electron_temperature_ev=tuple(temperature),
        ionization_source_current_a=tuple(source),
        electron_current_a=tuple(electron),
        ion_current_a=tuple(electron[4] - value for value in electron),
        cusp_ion_current_a=(0.0, 0.0, 0.0),
    )
    first = run_plasma_case(inputs, initial_states=(state, state))
    second = run_plasma_case(inputs, initial_states=(state, state))
    assert first == second
    serialized = serialize_plasma(first)
    assert serialized["valid_state"]
    assert serialized["residual_floor"] < 1.0e-15
    assert serialized["rank_status"]["jacobian_rank"] == 22
    assert all(
        len(attempt["residual_rows"]) == 28 for attempt in serialized["attempts"]
    )
