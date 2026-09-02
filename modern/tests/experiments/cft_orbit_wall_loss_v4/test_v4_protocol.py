"""Campaign protocol tests for cft_orbit_wall_loss_v4 (ported from v3 and extended)."""

from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import cft_revival.orbit_mc as orbit_mc
from cft_revival.experiment_runtime import canonical_bytes
from cft_revival.experiment_runtime.canonical import (
    CanonicalizationError,
    strict_json_file,
)
from cft_revival.orbit_mc import (
    AnalyticField,
    ElectronLaunch,
    EnsembleSummary,
    OrbitConfig,
    wilson_interval,
)
from cft_revival.orbit_mc.integrator import integrate_orbit

from experiments.cft_orbit_wall_loss_v4 import experiment
from experiments.cft_orbit_wall_loss_v4.experiment import (
    CASE_AUTHORITIES_PATH,
    EXPERIMENT,
    MODERN,
    ROLES,
    TIMESTEPS,
    _convergence,
    _final_velocity_equals_event_velocity,
    batch_records,
    build_all_case_authorities,
    build_case_launches,
    case_matrix,
    evidentiary_disjointness,
    evidentiary_plan,
    load_runtime_launch_payload,
    manufactured_gate_report,
    mu_diagnostic,
    orbit_mc_contract_report,
    orbit_mc_source_sha256,
    prior_design_launches,
    production_synthetic_preflight,
    protocol,
    require_orbit_mc_contract,
    result_diagnostics,
    runtime_launch_payload,
    schema,
    shakedown_plan,
)


def _mentions_mu(key: str) -> bool:
    """True for μ identifiers (mu_, _mu, magnetic_moment), not for 'maximum'."""

    parts = key.lower().split("_")
    return "mu" in parts or "magnetic_moment" in key.lower()


# The frozen contract binds the *executed* orbit_mc (1.6.0 at preregistration
# commit 757e365f). Before execution the live worktree had to match it exactly;
# after the terminal bundle exists the live package is allowed to move on (v1.7
# fixed sidecar EOL bytes) and the binding is checked against the recorded
# bundle and the preregistration commit's blobs instead.
FROZEN_ORBIT_MC_VERSION = "1.6.0"
RESULT_MANIFEST = experiment.RESULTS_ROOT / "manifest.json"
EXECUTED = RESULT_MANIFEST.is_file()


def _live_orbit_mc_matches_frozen_contract() -> bool:
    if not experiment.AUTHORITIES_PATH.is_file():
        return True
    authorities = strict_json_file(experiment.AUTHORITIES_PATH)
    return (
        orbit_mc.__version__ == authorities["orbit_mc_package_version"]
        and orbit_mc_source_sha256() == authorities["orbit_mc_source_sha256"]
    )


def _git_blob(commit: str, relative_to_repo: str) -> bytes:
    import subprocess

    return subprocess.run(
        ["git", "show", f"{commit}:{relative_to_repo}"],
        cwd=experiment.REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout


def _source_sha256_at_commit(commit: str, source_files: list[str]) -> str:
    """Recompute ``orbit_mc_source_sha256`` from the commit's LF blobs."""

    digest = hashlib.sha256()
    for relative in source_files:
        data = _git_blob(commit, f"modern/{relative}")
        assert b"\r" not in data, relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def test_only_qualified_divergent_exit_design_is_authorized() -> None:
    value = protocol()
    assert value["schema_version"] == schema("protocol")
    assert value["experiment_id"] == "cft-orbit-wall-loss-v4"
    assert value["authority"]["design_id"] == "divergent-exit-stack"
    assert value["authority"]["required_qualification"] == "NUMERICAL_P2_QUALIFIED"
    assert set(value["authority"]["excluded_designs"]) == {
        "historical-envelope-baseline",
        "compact-high-gradient-stack",
    }
    assert value["authority"]["orbit_mc_commit"] == "3ab50ef5c31cfa45f2256ddba18dafa965010c7a"
    assert value["authority"]["base_commit"] == "3ab50ef5c31cfa45f2256ddba18dafa965010c7a"
    assert value["classification"].endswith("not_pic")
    assert value["publication_boundary"]["hardware_or_experimental_validation"] is False
    assert value["publication_boundary"]["shakedown_outcomes_are_not_evidence"] is True


def test_nine_cases_have_exact_prefixed_equal_weight_launch_authorities() -> None:
    value = protocol()
    plan = evidentiary_plan(value)
    all_ids: set[str] = set()
    all_seeds: set[int] = set()
    for role, timestep, campaign_id in case_matrix(plan):
        assert campaign_id == f"cft-orbit-wall-loss-v4:{role}:{timestep}"
        launches = build_case_launches(value, plan, role, timestep)
        assert len(launches) == 512
        assert all(item.launch_id.startswith(campaign_id + ":") for item in launches)
        assert len({item.launch_id for item in launches}) == 512
        assert len({item.seed_id for item in launches}) == 512
        assert {item.parallel_direction for item in launches} == {-1, 1}
        strata = Counter(
            (
                item.flux_surface_id.split("-r", 1)[0],
                item.kinetic_energy_ev,
                item.pitch_angle_rad,
                item.parallel_direction,
            )
            for item in launches
        )
        assert len(strata) == 32
        assert set(strata.values()) == {16}
        batches = batch_records(plan, launches)
        assert len(batches) == 8
        assert {
            entry["weight"]
            for batch in batches
            for entry in batch["launches"]
        } == {1.0 / 512}
        all_ids.update(item.launch_id for item in launches)
        all_seeds.update(item.seed_id for item in launches)
    assert len(all_ids) == 4608
    assert len(all_seeds) == 4608


def test_launch_payload_is_exact_bytes_and_typed_roundtrip_in_memory() -> None:
    value = protocol()
    plan = evidentiary_plan(value)
    authorities = build_all_case_authorities(value, plan)
    assert authorities["case_count"] == 9
    assert authorities["plan_kind"] == "evidentiary"
    for authority in authorities["cases"]:
        launches = build_case_launches(
            value, plan, authority["role"], authority["timestep"]
        )
        payload = canonical_bytes(
            runtime_launch_payload(authority["campaign_id"], launches)
        )
        assert (
            hashlib.sha256(payload).hexdigest()
            == authority["runtime_launch_payload_byte_sha256"]
        )
        assert load_runtime_launch_payload(
            payload, authority["campaign_id"]
        ) == tuple(sorted(launches, key=lambda item: item.launch_id))


def test_every_frozen_case_file_is_exact_bytes_when_prepared() -> None:
    if not CASE_AUTHORITIES_PATH.is_file():
        pytest.skip("prepare has not frozen case authorities yet (phase 1)")
    value = protocol()
    plan = evidentiary_plan(value)
    authorities = strict_json_file(CASE_AUTHORITIES_PATH)
    assert authorities == build_all_case_authorities(value, plan)
    for authority in authorities["cases"]:
        launches = build_case_launches(
            value, plan, authority["role"], authority["timestep"]
        )
        expected = canonical_bytes(
            runtime_launch_payload(authority["campaign_id"], launches)
        )
        actual = (EXPERIMENT / authority["launch_manifest_path"]).read_bytes()
        assert actual == expected
        assert (
            hashlib.sha256(actual).hexdigest()
            == authority["runtime_launch_payload_byte_sha256"]
        )


def test_all_checkpoint_chains_and_fresh_grid_preflight_pass() -> None:
    value = protocol()
    plan = evidentiary_plan(value)
    audit = production_synthetic_preflight(
        value, build_all_case_authorities(value, plan)
    )
    assert len(audit["covered_fields"]) == 11
    assert "event_witness_v1_6_event_velocity_and_midpoint_fields" in audit["covered_fields"]
    assert "failure_event_witness_v1_6_zero_vectors" in audit["covered_fields"]
    assert len(audit["case_checkpoint_chains"]) == 9
    assert all(item["passed"] for item in audit["case_checkpoint_chains"])
    overlap = audit["overlap_evidence"]
    assert overlap["proven"] is True
    assert overlap["evidentiary_unique_launch_ids"] == 4608
    assert overlap["evidentiary_unique_seed_ids"] == 4608
    assert overlap["evidentiary_unique_positions"] == 8
    assert set(overlap["reports"]) == {
        "against_v1",
        "against_v2",
        "against_v3",
        "against_shakedown",
    }
    for report in overlap["reports"].values():
        assert report["disjoint"] is True
        assert set(report["overlap_counts"].values()) == {0}
    if EXECUTED and not _live_orbit_mc_matches_frozen_contract():
        # Post-execution drift of the live package is the only admissible
        # failure; every numerical/coverage check must still pass.
        failed = {key for key, passed in audit["checks"].items() if not passed}
        assert failed == {"orbit_mc_contract_matches"}, failed
        assert audit["orbit_mc_contract"]["matches"] is False
        assert audit["orbit_mc_contract"]["expected"]["package_version"] == FROZEN_ORBIT_MC_VERSION
        assert audit["orbit_mc_contract"]["observed"]["package_version"] == orbit_mc.__version__
        assert audit["passed"] is False
    else:
        assert audit["passed"]
        assert all(audit["checks"].values())
        assert audit["orbit_mc_contract"]["matches"] is True
    with pytest.raises(CanonicalizationError, match="reserved"):
        canonical_bytes({"__cft_type__": "tuple", "items": []})


def test_evidentiary_design_is_fresh_relative_to_v1_v2_v3() -> None:
    value = protocol()
    for version in ("v1", "v2", "v3"):
        prior = prior_design_launches(value, version)
        assert len(prior) in (512, 4608)
    report = evidentiary_disjointness(value)
    assert report["proven"] is True
    v3 = value["prior_campaign_disclosure"]["v3"]["design"]
    assert v3["gyrophase_offset_rad"] == pytest.approx(0.19634954084936207)
    assert value["launches"]["gyrophase_offset_rad"] == pytest.approx(
        v3["gyrophase_offset_rad"] + 0.2617993877991494
    )
    v3_positions = {tuple(position) for _, position in v3["positions_m"]}
    v4_positions = {tuple(item["position_m"]) for item in value["launches"]["position_seeds"]}
    assert not (v3_positions & v4_positions)


def test_three_map_three_timestep_protocol_and_all_required_gates() -> None:
    value = protocol()
    assert set(value["field_adapter"]["maps"]) == set(ROLES)
    assert set(value["orbit"]["timestep_policies"]) == set(TIMESTEPS)
    assert value["gates"]["maximum_successive_probability_change"] == 0.01
    assert value["gates"]["maximum_relative_energy_error"] == 1e-10
    assert value["gates"]["minimum_helix_position_order"] == 1.8
    assert value["gates"]["minimum_varying_e_position_order"] == 1.8
    assert value["gates"]["maximum_mirror_point_relative_error"] == 0.03
    assert value["gates"]["maximum_wall_endpoint_error_m"] == 1e-8
    assert value["gates"]["maximum_cpu_cuda_relative_velocity_difference"] == 1e-11
    assert value["gates"]["minimum_certificate_dense_to_bound_ratio"] == 0.001
    assert value["gates"]["require_exact_authority_replay"] is True
    assert value["gates"]["require_campaign_preflight"] is True
    assert value["gates"]["require_final_velocity_equals_event_velocity"] is True
    # Magnetic-moment variation is a diagnostic, never a gate of any kind.
    assert not any(_mentions_mu(key) for key in value["gates"])
    mu = value["diagnostics"]["magnetic_moment_variation"]
    assert mu["binding"] is False
    assert mu["informational_gate"] is False
    assert mu["role"] == "diagnostic_only"
    assert value["execution"]["single_execution"] is True
    assert value["execution"]["no_patch_or_rerun"] is True
    assert value["execution"]["real_field_shakedown_before_prepare"] is True
    assert value["execution"]["git_common_lock"] == "cft-orbit-wall-loss-v4.execution.lock"


def test_prior_campaign_disclosure_lists_exact_failures_and_shakedown_rule() -> None:
    disclosure = protocol()["prior_campaign_disclosure"]
    assert disclosure["v1"]["terminal_state"] == "prebundle_failure"
    assert (
        disclosure["v1"]["primary_error_message"]
        == "launch manifest differs from preregistered authority"
    )
    assert disclosure["v2"]["terminal_state"] == "runtime_failure"
    assert (
        disclosure["v2"]["primary_error_message"]
        == "ordered launch/result/campaign identities are inconsistent"
    )
    assert disclosure["v3"]["terminal_state"] == "runtime_failure"
    assert (
        disclosure["v3"]["primary_error_message"]
        == "physical event witness requires a positive step"
    )
    for version in ("v1", "v2", "v3"):
        assert disclosure[version]["root_cause"]
        assert len(disclosure[version]["preregistration_commit"]) == 40
        assert len(disclosure[version]["result_commit"]) == 40
    assert "shakedown" in disclosure["shakedown_rule"]
    assert disclosure["v4_launch_grid_reused"] is False


def test_orbit_mc_contract_and_source_hash_are_bound_to_this_worktree() -> None:
    value = protocol()
    report = orbit_mc_contract_report(value)
    expected = {
        "package_version": FROZEN_ORBIT_MC_VERSION,
        "result_schema_version": "cft-revival-orbit-mc-result/1.6.0",
        "checkpoint_schema_version": "cft-revival-orbit-mc-checkpoint/1.6.0",
        "validation_protocol_schema_version": (
            "cft-revival-orbit-mc-validation-protocol/1.6.0"
        ),
        "handoff_schema_version": "cft-revival-orbit-mc-coupling-v4.2/1.3.0",
    }
    assert report["expected"] == expected
    if EXECUTED and not _live_orbit_mc_matches_frozen_contract():
        # The evidence was produced under the frozen contract; the live package
        # has since moved (schemas unchanged). Check the recorded bindings.
        recorded = strict_json_file(
            experiment.RESULTS_ROOT / "artifacts" / "orbit-mc-contract.json"
        )
        authorities = strict_json_file(experiment.AUTHORITIES_PATH)
        assert recorded["matches"] is True
        assert recorded["expected"] == recorded["observed"] == expected
        assert authorities["orbit_mc_schema_versions"] == expected
        assert authorities["orbit_mc_package_version"] == FROZEN_ORBIT_MC_VERSION
        assert recorded["source_sha256"] == authorities["orbit_mc_source_sha256"]
        assert recorded["source_files"] == report["source_files"]
        lock = strict_json_file(experiment.RESULTS_ROOT / "execution-lock.json")
        assert recorded["source_sha256"] == _source_sha256_at_commit(
            lock["commit"], recorded["source_files"]
        )
        # Only the package version may differ; every schema version is frozen.
        drift = {
            key for key in expected if report["observed"][key] != expected[key]
        }
        assert drift == {"package_version"}, report["observed"]
        assert report["observed"]["package_version"] == orbit_mc.__version__ != "1.6.0"
    else:
        assert report["matches"] is True, report
        assert report["observed"] == expected
        assert orbit_mc.__version__ == FROZEN_ORBIT_MC_VERSION
    assert report["source_sha256"] == orbit_mc_source_sha256()
    assert len(report["source_sha256"]) == 64
    assert any(item.endswith("orbit_mc/integrator.py") for item in report["source_files"])
    assert any(item.endswith("spec/orbit_mc/result-v1.schema.json") for item in report["source_files"])
    # The hash is defined over LF bytes: every scoped file must be CR-free on disk.
    for relative in report["source_files"]:
        assert b"\r" not in (MODERN / relative).read_bytes(), relative
    # The evidentiary and shakedown plans must never share a campaign prefix.
    assert evidentiary_plan(value).campaign_id_prefix != shakedown_plan(value).campaign_id_prefix


def test_contract_fails_closed_on_package_version_or_schema_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = protocol()
    drifted = copy.deepcopy(value)
    drifted["orbit_mc_contract"]["package_version"] = "1.5.0"
    with pytest.raises(ValueError, match="package version"):
        require_orbit_mc_contract(drifted)
    drifted = copy.deepcopy(value)
    drifted["orbit_mc_contract"]["result_schema_version"] = "cft-revival-orbit-mc-result/1.5.0"
    with pytest.raises(ValueError, match="differs from protocol"):
        require_orbit_mc_contract(drifted)
    monkeypatch.setattr(orbit_mc, "__version__", "1.7.0")
    with pytest.raises(ValueError, match="differs from protocol"):
        require_orbit_mc_contract(value)


def test_source_hash_fails_closed_on_crlf_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    crlf = tmp_path / "crlf.py"
    crlf.write_bytes(b"x = 1\r\n")
    monkeypatch.setattr(experiment, "orbit_mc_source_files", lambda: [crlf])
    monkeypatch.setattr(experiment, "MODERN", tmp_path)
    with pytest.raises(RuntimeError, match="CR bytes"):
        orbit_mc_source_sha256()


def test_v1_6_event_velocity_identity_and_mu_diagnostic_on_real_integrator() -> None:
    """One synthetic-field orbit: final velocity IS the witnessed event velocity."""

    field = AnalyticField(lambda _position: np.array([0.0, 0.0, 0.2]), None, 0.2)
    config = OrbitConfig(
        wall_radius_m=0.002, wall_z_min_m=0.001, wall_z_max_m=0.018,
        domain_radius_m=0.002, domain_z_min_m=0.001, domain_z_max_m=0.023,
        max_time_s=1e-8, max_path_m=0.03, max_steps=200_000,
        max_rotation_rad=0.16, event_tolerance_m=1e-9, maximum_gamma=20.0,
    )
    launch = ElectronLaunch(
        launch_id="test:ev", seed_id=1, flux_surface_id="s",
        position_m=(0.0012, 0.0, 0.003), kinetic_energy_ev=25.0,
        pitch_angle_rad=math.radians(20.0), gyrophase_rad=0.0, parallel_direction=1,
    )
    result = integrate_orbit(launch, field, config)
    assert _final_velocity_equals_event_velocity(result) is True
    witness = result.event_witness
    for key in ("event_velocity_m_per_s", "step_magnetic_midpoint_t", "step_electric_midpoint_v_per_m"):
        assert key in witness and len(witness[key]) == 3
    assert result.maximum_relative_energy_error <= 1e-10
    diagnostics = result_diagnostics([result])
    assert diagnostics["final_velocity_equals_event_velocity_count"] == 1
    mu = diagnostics["magnetic_moment_variation_diagnostic"]
    assert mu["role"] == "diagnostic_only"
    assert mu["binding"] is False
    assert set(mu) >= {"min", "median", "max", "count_above_0p1", "count_above_0p5"}
    summary = mu_diagnostic([result, result])
    assert summary["orbit_count_with_mu"] + summary["orbit_count_without_mu"] == 2


def _summary(ensemble_id: str, wall: int, trials: int = 64) -> EnsembleSummary:
    return EnsembleSummary(
        ensemble_id,
        trials,
        wilson_interval(wall, trials),
        wilson_interval(0, trials),
        wilson_interval(trials - wall, trials),
        wilson_interval(0, trials),
        (("domain_escape", trials - wall), ("wall_hit", wall)),
        "0" * 64,
    )


def test_convergence_report_handles_three_levels_without_raising() -> None:
    """Regression: v3 zipped ordered/ordered[1:] with strict=True and always raised."""

    value = protocol()
    summaries = {
        (role, step): _summary(f"{role}:{step}", 38 + index)
        for role in ROLES
        for index, step in enumerate(TIMESTEPS)
    }
    report = _convergence(summaries, value)
    assert len(report["timestep"]) == 3
    assert len(report["cross_map"]) == 3
    assert all(len(row["successive_changes"]) == 2 for row in report["timestep"])
    assert all(len(row["adjacent_wilson_overlap"]) == 2 for row in report["cross_map"])
    assert report["timestep_passed"] is False  # 1/64 > 0.01 successive change
    assert report["cross_map_passed"] is True


def test_manufactured_production_preflight_passes_without_outcomes() -> None:
    report = manufactured_gate_report(protocol())
    assert report["passed"]
    assert all(report["checks"].values())
