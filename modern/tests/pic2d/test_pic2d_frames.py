"""Frame recorder (v2.0): cadence, shapes, exact interval averaging, atomic files, resume continuity, hash binding."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import uniform_field_map
from cft_revival.pic2d.frames import (
    FRAME_SCHEMA, MAP_KEYS, PROFILE_KEYS, SCALAR_KEYS, FrameRecorder, FrameRecorderConfig, estimate_frame_bytes, frame_scalars,
    frames_manifest, interval_maps, list_frames, load_frames,
)
from cft_revival.pic2d.mcc import XenonCrossSections
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import ChannelGeometry, Grid2D, PIC2DValidationError
from cft_revival.pic2d.simulation import DiagnosticAccumulator
from experiments.pic2d_cft_steady_state_v1 import run as runner

PLUME_PROTOCOL = Path(__file__).resolve().parents[2] / "experiments" / "pic2d_cft_plume_v1" / "protocol.json"


def _masks():
    return build_mesh_masks(Grid2D(ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3), 12, 96))


def _tiny_protocol(cadence: int = 40) -> dict:
    protocol = deepcopy(runner.load_protocol(PLUME_PROTOCOL))
    protocol["geometry"]["body_dielectric_radius_m"] = 0.0045
    protocol["case"].update({"radial_cells": 48, "axial_cells": 144, "macro_weight": 6.0e5})
    protocol["numerics"].update({"dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 80,
                                 "averaging_window_steps": 160, "frame_recorder": {"cadence_steps": cadence, "precision": "float32"}})
    protocol["numerics"]["stability_reference"]["density_per_m3"] = 1.0e16
    protocol["operating_point"]["seed_plasma_density_per_m3"] = 5.0e15
    return protocol


def test_config_validation_and_alignment_contract():
    cfg = FrameRecorderConfig(20000)
    cfg.validate_alignment(sync_steps=200, checkpoint_every_steps=40000, window_steps=400000)
    assert cfg.to_dict() == {"cadence_steps": 20000, "precision": "float32", "schema": FRAME_SCHEMA}
    with pytest.raises(PIC2DValidationError):
        FrameRecorderConfig(0)
    with pytest.raises(PIC2DValidationError):
        FrameRecorderConfig(100, "float64")
    with pytest.raises(PIC2DValidationError, match="device_sync_steps"):
        FrameRecorderConfig(150).validate_alignment(sync_steps=200, checkpoint_every_steps=40000, window_steps=400000)
    with pytest.raises(PIC2DValidationError, match="checkpoint_every_steps"):
        FrameRecorderConfig(30000).validate_alignment(sync_steps=200, checkpoint_every_steps=40000, window_steps=400000)
    with pytest.raises(PIC2DValidationError, match="averaging_window_steps"):
        FrameRecorderConfig(40000).validate_alignment(sync_steps=200, checkpoint_every_steps=40000, window_steps=100000)
    # the real plume protocol declares the recorder and its size estimate stays inside the 1-2 GB envelope
    real = runner.load_protocol(PLUME_PROTOCOL)
    fc = runner.frame_recorder_config(real)
    assert fc == FrameRecorderConfig(20000, "float32")
    numerics = real["numerics"]
    fc.validate_alignment(sync_steps=numerics["device_sync_steps"], checkpoint_every_steps=numerics["checkpoint_every_steps"],
                          window_steps=numerics["averaging_window_steps"])
    per_frame = estimate_frame_bytes((241, 721), "float32")
    assert per_frame == 241 * 721 * 4 * 7 and per_frame * 257 < 2e9   # 257 frames = 7.7 us at 30 ns per frame
    assert runner.frame_recorder_config({"numerics": {}}) is None       # default OFF (v1.x protocols)


def test_interval_maps_are_exact_differences_of_the_window_sums():
    """Synthetic accumulator: the frame over [a, b] equals to_arrays of an accumulator holding only [a, b]."""

    masks = _masks()
    rng = np.random.default_rng(7)
    shape = masks.grid.node_shape
    nz, nr = masks.grid.axial_cells, masks.grid.radial_cells

    def synthetic(steps: int) -> DiagnosticAccumulator:
        acc = DiagnosticAccumulator(masks)
        acc.steps = steps
        for key in ("n_e", "n_i", "phi", "e_weight", "e_v2", "ionization"):
            setattr(acc, key, rng.random(shape) * steps * 1e17)
        for key in ("e_vr", "e_vt", "e_vz"):
            setattr(acc, key, (rng.random(shape) - 0.5) * steps * 1e22)
        acc.e_v2 = rng.random(shape) * steps * 1e29     # sum w v^2 well above the drift term: no cancellation in T_e
        acc.wall_ions = rng.random(nz) * steps
        acc.wall_electrons = rng.random(nz) * steps
        acc.exit_ions = rng.random(nr) * steps
        acc.theta_ions = rng.random(90) * steps
        acc.iedf_ions = rng.random(256) * steps
        return acc

    first = synthetic(300)      # accumulated over [0, 300]
    second = synthetic(200)     # accumulated over [300, 500]
    cumulative = DiagnosticAccumulator(masks)
    for key in DiagnosticAccumulator.SUM_KEYS:
        setattr(cumulative, key, getattr(first, key) + getattr(second, key))
    cumulative.steps = 500
    maps = interval_maps(cumulative.raw_sums(), first.raw_sums(), masks, 6e4, 1.5e-12)
    expected = second.to_arrays(6e4, 1.5e-12)
    assert int(maps["window_steps"][0]) == 200
    for key in ("n_e_per_m3", "n_i_per_m3", "phi_v", "ionization_rate_per_m3_s", "sample_count_e", "wall_ion_flux_per_m2_s",
                "exit_ion_current_density_a_per_m2", "plume_ion_current_per_sr_a", "iedf_ion_counts"):
        np.testing.assert_allclose(maps[key], expected[key], rtol=1e-9, atol=0.0, err_msg=key)
    # T_e from the moments: the drift correction needs the raw sums, which is why the recorder differences sums, not maps
    np.testing.assert_allclose(maps["t_e_ev"], expected["t_e_ev"], rtol=1e-6, atol=1e-9)
    # the first frame of a window has no previous snapshot: it is the window accumulator itself
    np.testing.assert_array_equal(interval_maps(first.raw_sums(), None, masks, 6e4, 1.5e-12)["n_e_per_m3"], first.to_arrays(6e4, 1.5e-12)["n_e_per_m3"])
    with pytest.raises(PIC2DValidationError):
        interval_maps(first.raw_sums(), first.raw_sums(), masks, 6e4, 1.5e-12)
    # scalars flatten the series record and tolerate missing blocks
    flat = frame_scalars({"step": 3, "time_s": 1e-9, "electrons": 5, "ions": 6, "currents_a": {"discharge_a": 0.01}})
    assert set(flat) == set(SCALAR_KEYS) and flat["discharge_a"] == 0.01 and flat["thrust_total_n"] is None
    assert frame_scalars(None)["step"] is None


def test_runner_records_frames_at_the_cadence_and_resume_is_continuous(tmp_path: Path):
    protocol = _tiny_protocol(cadence=40)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    config = runner.build_config(protocol, backend="cpu")
    field = uniform_field_map(config.grid, 0.02)
    xs = XenonCrossSections.from_file()
    results = tmp_path / "run"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=200, protocol_path=protocol_path,
                            log=lambda _: None)
    files = list_frames(results)
    assert [p.name for p in files] == [f"frame-{i:06d}.npz" for i in range(5)]     # 200 / 40
    frames = load_frames(results)
    assert frames.count == 5 and frames.precision == "float32" and frames.schema == FRAME_SCHEMA
    np.testing.assert_array_equal(frames.start_step, [0, 40, 80, 120, 160])
    np.testing.assert_array_equal(frames.end_step, [40, 80, 120, 160, 200])
    np.testing.assert_allclose(frames.time_s, frames.end_step * 5e-12, rtol=1e-12)
    node_shape = tuple(config.grid.node_shape)
    for key in MAP_KEYS:
        assert frames.maps[key].shape == (5, *node_shape) and frames.maps[key].dtype == np.float32, key
    assert frames.surface_charge_c.shape == (5, *node_shape)
    assert frames.profiles["wall_ion_flux_per_m2_s"].shape == (5, config.grid.axial_cells)
    assert frames.profiles["plume_ion_current_per_sr_a"].shape == (5, 90) and frames.profiles["iedf_ion_counts"].shape == (5, 256)
    assert set(frames.scalars) == set(SCALAR_KEYS) and np.all(frames.scalars["electrons"] > 0) and np.all(np.isfinite(frames.scalars["thrust_total_n"]))
    assert np.all(frames.scalars["step"] == frames.end_step)
    # the electron sample counts are positive in the seeded channel and the maps are finite where sampled
    assert frames.maps["sample_count_e"][0, :4, :20].sum() > 0
    assert np.all(np.isfinite(frames.maps["n_e_per_m3"]))
    # frames 0-3 (window [0, 160]) sum to the window mean written at the reset; the 160-step window = 4 frames
    weights = np.diff(np.concatenate([[0], frames.end_step[:4]])).astype(np.float64)
    combined = np.tensordot(weights, frames.maps["n_e_per_m3"][:4].astype(np.float64), axes=(0, 0)) / weights.sum()
    summary = artifacts.read_canonical_json(results / "summary.json")
    with np.load(results / "maps.npz") as maps:
        if summary["averaging_window_step_range"] == [0, 160]:
            np.testing.assert_allclose(combined, maps["n_e_per_m3"], rtol=2e-6, atol=1e6)
    manifest = summary["artifacts"]["frames"]
    assert manifest["count"] == 5 and manifest["first_end_step"] == 40 and manifest["last_end_step"] == 200
    assert manifest["config"] == {"cadence_steps": 40, "precision": "float32", "schema": FRAME_SCHEMA}
    assert manifest["sha256"] == frames_manifest(results)["sha256"] and len(manifest["sha256"]) == 64
    digests_before = {p.name: p.read_bytes() for p in files}

    # kill-between-frame-and-checkpoint simulation: a stray frame past the checkpoint is removed on resume, none duplicated
    stray = results / "frames" / "frame-000005.npz"
    with np.load(files[-1]) as last:
        payload = {k: last[k] for k in last.files}
    payload["start_step"] = np.array([200]); payload["end_step"] = np.array([240])
    np.savez_compressed(stray, **payload)
    assert len(list_frames(results)) == 6
    runner.run_steady_state(protocol, results, backend="cpu", field_map=field, cross_sections=xs, max_steps=320, protocol_path=protocol_path,
                            log=lambda _: None)
    resumed = load_frames(results)
    np.testing.assert_array_equal(resumed.end_step, [40, 80, 120, 160, 200, 240, 280, 320])
    np.testing.assert_array_equal(resumed.start_step[1:], resumed.end_step[:-1])
    for name, blob in digests_before.items():             # the first session's frames are untouched by the resume
        assert (results / "frames" / name).read_bytes() == blob
    summary2 = artifacts.read_canonical_json(results / "summary.json")
    assert summary2["artifacts"]["frames"]["count"] == 8 and summary2["steps_completed"] == 320
    # deterministic: the same run from scratch reproduces every frame byte for byte
    again = tmp_path / "again"
    runner.run_steady_state(protocol, again, backend="cpu", field_map=field, cross_sections=xs, max_steps=200, protocol_path=protocol_path,
                            log=lambda _: None)
    for a, b in zip(list_frames(results)[:5], list_frames(again)):
        with np.load(a) as fa, np.load(b) as fb:
            for key in fa.files:
                np.testing.assert_array_equal(fa[key], fb[key], err_msg=key)
    # a v1.x protocol (no frame_recorder block) writes no frames and reports None
    protocol_off = deepcopy(protocol)
    protocol_off["numerics"].pop("frame_recorder")
    off = tmp_path / "off"
    runner.run_steady_state(protocol_off, off, backend="cpu", field_map=field, cross_sections=xs, max_steps=80, log=lambda _: None)
    assert list_frames(off) == [] and artifacts.read_canonical_json(off / "summary.json")["artifacts"]["frames"] is None
