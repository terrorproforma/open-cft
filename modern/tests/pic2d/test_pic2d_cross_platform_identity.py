"""Cross-platform replay policy of the PIC-2D identities (bitwise on the anchor platform, tolerance-gated elsewhere).

The sampled node field carries two identities: ``sha256`` (content hash of the sampled arrays + full provenance, a
bitwise identity on the platform that produced it) and ``source_sha256`` (grid + declared file-byte hashes of the P2
bundle, platform-independent).  A checkpoint binds both plus an anchor copy of the arrays, so a resume on another
CPU / BLAS / OS verifies the SAME source and gates the re-sampled map against the anchor under the declared ULP-scale
tolerance instead of refusing on a last-digit content-hash difference; the replay mode is recorded with the session.
"""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import (
    DERIVED_PROVENANCE_KEYS,
    FIELD_REPLAY_ATOL_OVER_MAX_B,
    FIELD_REPLAY_RTOL,
    MagneticFieldMap,
    compare_field_arrays,
    linear_psi_field_map,
    source_provenance,
)
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.models import BoundaryPotentials, ChannelGeometry, Grid2D, PIC2DValidationError, PoissonConfig2D, StabilityLimits
from cft_revival.pic2d.simulation import InjectionConfig, PIC2DConfig, SeedPlasmaConfig, Simulation
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v1.gpu_sampler import GpuUtilisationSampler


@pytest.fixture(autouse=True)
def _no_nvidia_smi(monkeypatch):
    """The runner's background GPU sampler must not spawn nvidia-smi in the test suite (hermetic, GPU may be busy)."""

    monkeypatch.setattr(runner, "GpuUtilisationSampler", functools.partial(GpuUtilisationSampler, query=lambda timeout_s: None))


GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
DECLARED = {
    "kind": "p2-psi-bicubic-node-sample", "role": "primary", "design_id": "divergent-exit-stack",
    "checkpoint_path": "modern/examples/x.json", "checkpoint_file_sha256": "a" * 64, "checkpoint_payload_sha256": "b" * 64,
    "checkpoint_sidecar_sha256": "c" * 64, "mesh_sha256": "d" * 64, "run_sha256": "e" * 64, "authority_file_sha256": "f" * 64,
    "source_identity_sha256": "1" * 64, "psi_grid": {"radial_samples": 41, "axial_samples": 481, "r_min_m": 0.0, "r_max_m": 0.004},
    "sampling_regions": ["channel-straight"],
}
DERIVED = {
    "withheld_midcell_error": {"sample_count": 100, "b_rms_t": 1.2e-4, "b_relative_rms": 3.1e-4},
    "node_reference_b_max_abs_error_t": 2.0e-5, "certified_max_b_t": 0.41234567890123,
    "certificate": {"ratio": 1.0000000001}, "channel_cross_check": {"channel_field_map_sha256": "9" * 64, "max_abs_diff_t": 1.06e-3},
}


def _p2_like_map(grid: Grid2D, *, derived: dict = DERIVED, declared: dict = DECLARED, ulp_shift: int = 0) -> MagneticFieldMap:
    base = linear_psi_field_map(grid, 2.0)
    b_z = base.b_z_t.copy()
    for _ in range(ulp_shift):
        b_z = np.nextafter(b_z, np.inf)
    return MagneticFieldMap(grid, base.b_r_t.copy(), b_z, {**declared, **derived})


def _config(grid: Grid2D) -> PIC2DConfig:
    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e5, seed=7,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e15, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="direct", relative_tolerance=1e-10), limits=StabilityLimits(max_cell_debye_ratio=4.0),
        reference_density_per_m3=1e15, reference_electron_temperature_ev=5.0, series_interval_steps=10, runtime_stability_check_steps=10,
    )


# -- the two identities of a field map -----------------------------------------------------------------------------

def test_source_identity_binds_declared_inputs_only():
    grid = Grid2D(GEOMETRY, 12, 96)
    anchor = _p2_like_map(grid)
    shifted = _p2_like_map(grid, ulp_shift=1)                      # another CPU: last-digit differences of the sample
    other_derived = _p2_like_map(grid, derived={**DERIVED, "channel_cross_check": {"channel_field_map_sha256": "8" * 64, "max_abs_diff_t": 1.07e-3}})
    other_source = _p2_like_map(grid, declared={**DECLARED, "checkpoint_file_sha256": "0" * 64})
    other_grid = _p2_like_map(Grid2D(GEOMETRY, 12, 48))
    assert anchor.sha256 != shifted.sha256 and anchor.source_sha256 == shifted.source_sha256
    assert anchor.sha256 != other_derived.sha256 and anchor.source_sha256 == other_derived.source_sha256
    assert anchor.source_sha256 != other_source.source_sha256 and anchor.source_sha256 != other_grid.source_sha256
    assert set(source_provenance(anchor.provenance)) == set(DECLARED) and DERIVED_PROVENANCE_KEYS >= set(DERIVED)
    record = anchor.to_dict()
    assert record["field_map_sha256"] == anchor.sha256 and record["field_source_sha256"] == anchor.source_sha256
    # analytic maps: the declared inputs ARE the provenance
    analytic = linear_psi_field_map(grid, 2.0)
    assert analytic.source_sha256 == linear_psi_field_map(grid, 2.0).source_sha256 != linear_psi_field_map(grid, 2.5).source_sha256


def test_compare_field_arrays_declares_an_ulp_scale_tolerance():
    grid = Grid2D(GEOMETRY, 12, 96)
    anchor = _p2_like_map(grid)
    same = compare_field_arrays(anchor, anchor.b_r_t, anchor.b_z_t)
    assert same["bitwise"] and same["within_tolerance"] and same["max_abs_diff_t"] == 0.0 and same["nodes_differing"] == 0
    assert same["rtol"] == FIELD_REPLAY_RTOL == 1e-12 and same["atol_over_max_b"] == FIELD_REPLAY_ATOL_OVER_MAX_B == 1e-12
    assert same["atol_t"] == pytest.approx(1e-12 * anchor.max_b_t)
    shifted = _p2_like_map(grid, ulp_shift=3)
    near = compare_field_arrays(shifted, anchor.b_r_t, anchor.b_z_t)
    assert not near["bitwise"] and near["within_tolerance"] and 0 < near["max_rel_diff"] < 1e-14 and near["nodes_differing"] > 0
    far = MagneticFieldMap(grid, anchor.b_r_t, anchor.b_z_t * (1 + 1e-9), anchor.provenance)
    beyond = compare_field_arrays(far, anchor.b_r_t, anchor.b_z_t)
    assert not beyond["within_tolerance"] and beyond["max_rel_diff"] == pytest.approx(1e-9, rel=1e-3)
    with pytest.raises(PIC2DValidationError, match="anchor field arrays"):
        compare_field_arrays(anchor, anchor.b_r_t[:, :-1], anchor.b_z_t)


# -- checkpoint binding: source identity + bitwise / numerical / refused ---------------------------------------------

def test_checkpoint_field_binding_modes(tmp_path: Path):
    grid = Grid2D(GEOMETRY, 12, 96)
    config = _config(grid)
    xs = XenonCrossSections.from_file()
    anchor = _p2_like_map(grid)
    sim = Simulation(config, anchor, cross_sections=xs)
    sim.run(20)
    json_path, npz_path = artifacts.save_checkpoint(tmp_path, "ckpt", sim.state, config, field_sha256=anchor.sha256, field=anchor,
                                                    cross_section_sha256=xs.payload_sha256, backend="cpu")
    metadata = artifacts.read_canonical_json(json_path)
    assert metadata["field_sha256"] == anchor.sha256 and metadata["field_source_sha256"] == anchor.source_sha256
    assert metadata["field_anchor_file"] == "ckpt.field.npz" and (tmp_path / "ckpt.field.npz").is_file() and (tmp_path / "ckpt.field.npz.sha256.json").is_file()
    anchor_arrays = artifacts.read_npz(tmp_path / "ckpt.field.npz", expected_sha256=metadata["field_anchor_sha256"])
    assert np.array_equal(anchor_arrays["b_z_t"], anchor.b_z_t) and "rtol 1e-12" in metadata["field_replay_policy"]
    assert metadata["runtime"]["platform_fingerprint"]["fingerprint_sha256"] == artifacts.platform_fingerprint()["fingerprint_sha256"]
    # the state arrays file is the same bytes with or without the anchor option (finalize --recover-runner-stop re-hashes it)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_json, legacy_npz = artifacts.save_checkpoint(legacy_dir, "ckpt", sim.state, config, field_sha256=anchor.sha256,
                                                        cross_section_sha256=xs.payload_sha256, backend="cpu")
    assert legacy_npz.read_bytes() == npz_path.read_bytes() and not (legacy_dir / "ckpt.field.npz").exists()
    # same platform: bitwise
    report: dict = {}
    state = artifacts.load_checkpoint(json_path, config, field_sha256=anchor.sha256, field=anchor, cross_section_sha256=xs.payload_sha256,
                                      identity_report=report)
    assert state.step == 20 and report["field"]["mode"] == "bitwise" and report["field"]["field_source_sha256"] == anchor.source_sha256
    # another CPU / BLAS: the re-sampled map differs in its last digits -> numerical replay, recorded as such
    shifted = _p2_like_map(grid, ulp_shift=2)
    report = {}
    artifacts.load_checkpoint(json_path, config, field_sha256=shifted.sha256, field=shifted, cross_section_sha256=xs.payload_sha256,
                              identity_report=report)
    numerical = report["field"]
    assert numerical["mode"] == "numerical" and numerical["anchor_field_sha256"] == anchor.sha256 and numerical["live_field_sha256"] == shifted.sha256
    assert numerical["comparison"]["within_tolerance"] and not numerical["comparison"]["bitwise"] and numerical["comparison"]["max_rel_diff"] < 1e-14
    assert numerical["anchor_platform_fingerprint"] == numerical["live_platform_fingerprint"] == artifacts.platform_fingerprint()["fingerprint_sha256"]
    # beyond the declared tolerance: refused (same source, but not the same field)
    far = MagneticFieldMap(grid, anchor.b_r_t, anchor.b_z_t * (1 + 1e-9), anchor.provenance)
    with pytest.raises(PIC2DValidationError, match="beyond the declared cross-platform tolerance"):
        artifacts.load_checkpoint(json_path, config, field_sha256=far.sha256, field=far, cross_section_sha256=xs.payload_sha256)
    # a different P2 bundle (source identity) is refused before any array comparison, even with identical arrays
    other_source = MagneticFieldMap(grid, anchor.b_r_t, anchor.b_z_t, {**anchor.provenance, "checkpoint_file_sha256": "0" * 64})
    with pytest.raises(PIC2DValidationError, match="source identity differs"):
        artifacts.load_checkpoint(json_path, config, field_sha256=other_source.sha256, field=other_source, cross_section_sha256=xs.payload_sha256)
    # the content hash passed must belong to the live map (no mix-and-match)
    with pytest.raises(PIC2DValidationError, match="does not belong"):
        artifacts.load_checkpoint(json_path, config, field_sha256=anchor.sha256, field=shifted, cross_section_sha256=xs.payload_sha256)
    with pytest.raises(PIC2DValidationError, match="does not belong"):
        artifacts.save_checkpoint(tmp_path / "bad", "x", sim.state, config, field_sha256=shifted.sha256, field=anchor,
                                  cross_section_sha256=xs.payload_sha256, backend="cpu")
    # without the live map (legacy call) the content hash must match exactly - the old fail-closed behaviour
    with pytest.raises(PIC2DValidationError, match="field identity differs"):
        artifacts.load_checkpoint(json_path, config, field_sha256=shifted.sha256, cross_section_sha256=xs.payload_sha256)
    # a legacy checkpoint (no source identity / anchor recorded) admits only a bitwise field, whatever is passed
    report = {}
    artifacts.load_checkpoint(legacy_json, config, field_sha256=anchor.sha256, field=anchor, cross_section_sha256=xs.payload_sha256, identity_report=report)
    assert report["field"]["mode"] == "bitwise" and "legacy" in report["field"]["basis"]
    with pytest.raises(PIC2DValidationError, match="field identity differs"):
        artifacts.load_checkpoint(legacy_json, config, field_sha256=shifted.sha256, field=shifted, cross_section_sha256=xs.payload_sha256)
    # a checkpoint whose anchor arrays are missing cannot admit a numerical replay
    (tmp_path / "ckpt.field.npz").unlink()
    with pytest.raises(PIC2DValidationError, match="no anchor arrays"):
        artifacts.load_checkpoint(json_path, config, field_sha256=shifted.sha256, field=shifted, cross_section_sha256=xs.payload_sha256)


# -- runtime identity: where was this produced ---------------------------------------------------------------------

def test_runtime_identity_records_the_cpu_platform_and_the_gpu():
    fingerprint = artifacts.platform_fingerprint()
    assert fingerprint["schema"] == artifacts.PLATFORM_FINGERPRINT_SCHEMA and fingerprint is artifacts.platform_fingerprint()
    for key in (*artifacts.PLATFORM_FINGERPRINT_KEYS, "os_release", "python", "cpu_model", "fingerprint_sha256"):
        assert key in fingerprint, key
    assert fingerprint["os"] and fingerprint["machine"] and fingerprint["numpy"] == np.__version__
    assert isinstance(fingerprint["simd_baseline"], list) and isinstance(fingerprint["simd_dispatch_enabled"], list)
    assert set(fingerprint["blas"]) == {"name", "version", "build_configuration", "runtime_corename"}
    # the fingerprint is the content hash of its declared determinants only (CPU model / Python patch level are informational)
    from cft_revival.orbit_mc.artifacts import content_hash

    assert fingerprint["fingerprint_sha256"] == content_hash({key: fingerprint[key] for key in artifacts.PLATFORM_FINGERPRINT_KEYS})
    assert "cpu_model" not in artifacts.PLATFORM_FINGERPRINT_KEYS and "python" not in artifacts.PLATFORM_FINGERPRINT_KEYS
    runtime = artifacts.runtime_identity()
    assert runtime["platform_fingerprint"] == fingerprint and runtime["code_sha256"] == artifacts.code_identity()
    assert "gpu" in runtime and (runtime["gpu"] is None or {"driver_version", "toolkit_version"} <= set(runtime["gpu"]))
    json.dumps(runtime, allow_nan=False)      # canonical-JSON safe (no NaN, no numpy scalars)


def test_gpu_identity_is_read_from_an_initialised_warp_runtime_only():
    pytest.importorskip("warp")
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    wp.init()
    gpu = artifacts.gpu_identity()
    assert gpu is not None and gpu["name"] and isinstance(gpu["arch"], int) and gpu["total_memory_bytes"] > 0
    assert gpu["driver_version"] and gpu["toolkit_version"]
    assert artifacts.runtime_identity()["gpu"] == gpu


# -- the runner records the replay mode with the session -------------------------------------------------------------

def _tiny_protocol() -> dict:
    """The v1 steady-state protocol shrunk to a 12 x 96 CPU run (as tests/pic2d/test_pic2d_steady_state_runner.py)."""

    protocol = copy.deepcopy(runner.load_protocol())
    protocol["case"].update({"radial_cells": 12, "axial_cells": 96, "macro_weight": 2.0e6, "seed": 3})
    protocol["operating_point"].update({"neutral_density_per_m3": 1.0e21, "electron_injection_current_a": 0.05, "seed_plasma_density_per_m3": 1.0e16})
    protocol["numerics"].update({"dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 40,
                                 "averaging_window_steps": 80, "ion_subcycle": 1})
    protocol["numerics"]["stability_limits"]["max_cell_debye_ratio"] = 4.0
    protocol["numerics"]["stability_reference"] = {"density_per_m3": 1.0e16, "electron_temperature_ev": 5.0, "max_electron_energy_ev": 400.0}
    protocol["budget_v1_2"].update({"ion_transit_time_s": 1.0e-9, "n_max_per_m3": 4.0e17, "n_eq_projected_per_m3": 1.0e17})
    return protocol


def test_runner_resume_records_the_field_replay_mode(tmp_path: Path):
    protocol = _tiny_protocol()
    config = runner.build_config(protocol, backend="cpu")
    field = _p2_like_map(config.grid)
    xs = XenonCrossSections.from_file()
    results = tmp_path / "results"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=80, log=lambda _: None)
    checkpoint = artifacts.read_canonical_json(results / "checkpoint" / "checkpoint-latest.json")
    assert checkpoint["field_source_sha256"] == field.source_sha256 and (results / "checkpoint" / "checkpoint-latest.field.npz").is_file()
    # resume with the map re-sampled on "another CPU" (last-digit differences): admitted as a numerical replay and recorded
    shifted = _p2_like_map(config.grid, ulp_shift=1)
    logs: list[str] = []
    runner.run_steady_state(protocol, results, backend="cpu", field_map=shifted, cross_sections=xs, max_steps=120, log=logs.append)
    state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
    assert [s.get("field_identity", {}).get("mode") for s in state["sessions"]] == [None, "numerical"]
    assert state["sessions"][1]["field_identity"]["comparison"]["within_tolerance"] and any("field replay numerical" in line for line in logs)
    summary = artifacts.read_canonical_json(results / "summary.json")
    assert summary["sessions"][1]["field_identity"]["anchor_field_sha256"] == field.sha256
    assert summary["provenance"]["field"]["field_source_sha256"] == field.source_sha256 == shifted.source_sha256
    assert summary["provenance"]["runtime"]["platform_fingerprint"]["fingerprint_sha256"] == artifacts.platform_fingerprint()["fingerprint_sha256"]
    # the same map resumes bitwise
    logs.clear()
    runner.run_steady_state(protocol, results, backend="cpu", field_map=shifted, cross_sections=xs, max_steps=160, log=logs.append)
    state = json.loads((results / "run_state.json").read_text(encoding="utf-8"))
    assert state["sessions"][2]["field_identity"]["mode"] == "bitwise" and any("field replay bitwise" in line for line in logs)
    # a map beyond the tolerance refuses the resume (fail closed)
    far = MagneticFieldMap(config.grid, field.b_r_t, field.b_z_t * (1 + 1e-9), field.provenance)
    with pytest.raises(PIC2DValidationError, match="beyond the declared cross-platform tolerance"):
        runner.run_steady_state(protocol, results, backend="cpu", field_map=far, cross_sections=xs, max_steps=200, log=lambda _: None)
