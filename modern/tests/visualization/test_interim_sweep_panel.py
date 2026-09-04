"""Interim sweep panel: synthesised summary, tolerant frame staging, status series, comparison PNGs, banner videos, players."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cft_revival.pic2d.frames import FRAME_SCHEMA, MAP_KEYS, SCALAR_KEYS, load_frames
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import ChannelGeometry, Grid2D

MODERN = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("interim_sweep_panel", MODERN / "visualization" / "interim_sweep_panel.py")
panel = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = panel          # dataclasses resolve string annotations through sys.modules[cls.__module__]
spec.loader.exec_module(panel)
video = panel.video

W_MACRO = 6.0e4
DT_STEP = 1.5e-12
FRAME_STEPS = 20000


def _synthetic_run(root: Path, design_id: str, *, radial_cells: int = 8, axial_cells: int = 24, n_frames: int = 5, seed: int = 1,
                   truncated: bool = True, density_scale: float = 1.0) -> Path:
    """A results directory as the runner leaves it while running: protocol.json, frames/, status.jsonl, run_state.json (no summary.json)."""

    cell = 5.0e-5
    results = root / design_id / "results"
    (results / "frames").mkdir(parents=True)
    length, radius = axial_cells * cell, radial_cells * cell
    geometry = {"bore_radius_m": radius, "z_min_m": 0.0, "z_max_m": length, "cone_start_z_m": length, "exit_radius_m": radius}
    protocol = {"design_id": design_id, "experiment_id": "synthetic-sweep", "model_version": "synthetic", "status": "synthetic_test_protocol",
                "design": {"note": "synthetic"}, "case": {"radial_cells": radial_cells, "axial_cells": axial_cells, "macro_weight": W_MACRO},
                "numerics": {"dt_s": DT_STEP}, "geometry": geometry}
    (results / "protocol.json").write_text(json.dumps(protocol, indent=1), encoding="utf-8")
    grid = Grid2D(ChannelGeometry(radius, 0.0, length, length, radius), radial_cells, axial_cells)
    masks = build_mesh_masks(grid)
    plasma, volume = masks.plasma_node, np.asarray(masks.shape_volume_m3, dtype=np.float64)
    nr, nz = plasma.shape
    z = np.arange(nz) * cell
    rng = np.random.default_rng(seed)
    true_rate = np.zeros((nr, nz))
    true_rate[1:nr - 2, nz // 3:2 * nz // 3] = 6.0e24            # one ionising band in the middle third of the channel
    records = []
    for i in range(n_frames):
        start, end = i * FRAME_STEPS, (i + 1) * FRAME_STEPS
        dt_frame = FRAME_STEPS * DT_STEP
        counts = rng.poisson(true_rate * volume * dt_frame / W_MACRO).astype(np.float64)
        counts[~plasma] = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(plasma, counts * W_MACRO / (volume * dt_frame), 0.0)
        n_i = np.where(plasma, density_scale * 2.0e17 * (1.0 + 0.2 * i) * np.exp(-((z - length / 2) / (length / 4)) ** 2)[None, :], 0.0)
        maps = {"n_e_per_m3": n_i * 0.97, "n_i_per_m3": n_i, "phi_v": np.where(plasma, 300.0 * (1.0 - z / length)[None, :], 0.0),
                "t_e_ev": np.where(plasma, 6.0, 0.0), "ionization_rate_per_m3_s": rate, "sample_count_e": np.where(plasma, 60.0, 0.0)}
        scalars = {key: None for key in SCALAR_KEYS}
        scalars.update({"step": end, "time_s": end * DT_STEP, "electrons": 1.0e5 * (i + 1), "ions": 1.1e5 * (i + 1), "discharge_a": 1.0e-3 * (1 + 0.1 * i),
                        "exit_ion_beam_a": 2.0e-4, "ionization_rate_per_s": float((rate * volume).sum()), "neutral_density_per_m3": 4.0e19})
        payload = {"schema": np.array([FRAME_SCHEMA]), "precision": np.array(["float32"]), "start_step": np.array([start], dtype=np.int64),
                   "end_step": np.array([end], dtype=np.int64), "interval_steps": np.array([FRAME_STEPS], dtype=np.int64),
                   "time_s": np.array([end * DT_STEP]), "scalars_json": np.array([json.dumps(scalars, sort_keys=True)]),
                   "surface_charge_c": np.zeros(plasma.shape, dtype=np.float32)}
        payload.update({key: np.asarray(maps[key], dtype=np.float32) for key in MAP_KEYS})
        with (results / "frames" / f"frame-{i:06d}.npz").open("wb") as handle:
            np.savez_compressed(handle, **payload)
        for step in range(start + 200, end + 1, 200):
            records.append({"step": step, "time_s": step * DT_STEP, "discharge_a": 1.0e-3 * (1 + 0.1 * i), "ionization_rate_per_s": 1.0e16 * (1 + i),
                            "electrons": 1.0e5 * (i + 1), "ions": 1.1e5 * (i + 1), "n_e_peak_node_per_m3": 8.0e17, "n_g_per_m3": 4.0e19, "ms_per_step": 4.0 + 0.1 * i,
                            "t_e_mean_ev": 8.0, "wall_seconds_total": step * 0.004,
                            "peak_node": {"cells_per_debye": 1.1 + 0.01 * i, "n_e_peak_per_m3": 4.0e17, "window": {"cells_per_debye": 0.6, "window_complete": False}},
                            "grid_heating_triad": {"windowed_energy_residual_over_electrode_work": -0.08, "windowed_energy_residual_window_complete": False,
                                                   "soft_ok": True, "hard_failures": []},
                            "plateau": {"reached": False}})
    if truncated:
        (results / "frames" / f"frame-{n_frames:06d}.npz").write_bytes(b"PK\x03\x04 truncated mid-write")
    lines = [json.dumps(r) for r in records] + ['{"step": 999999, "time_s": 1e-6, "discharge_a": 0.0']      # a partial last line
    (results / "status.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (results / "run_state.json").write_text(json.dumps({"checkpoint_step": n_frames * FRAME_STEPS, "finished": False, "frames_written": n_frames}), encoding="utf-8")
    return results


def _manifest(tmp_path: Path, a: Path, b: Path) -> Path:
    length_a, length_b = 24 * 5.0e-5, 30 * 5.0e-5
    runs = [
        {"job_id": "job-hot", "design_id": "synthetic-hot", "label": "hot", "results": str(b), "rho": 2.3, "rho_source": "test", "r_w_over_l": 0.9,
         "cusp_z_m": [length_b / 3, 2 * length_b / 3], "boundary_cusp_z_m": [], "cusp_source": "test", "transit_time_s": 1.7e-6, "target_transits": 3.0},
        {"job_id": "job-cold", "design_id": "synthetic-cold", "label": "cold", "results": str(a), "rho": 0.4, "rho_source": "test", "r_w_over_l": 0.24,
         "cusp_z_m": [1.0e-4, length_a / 3, 2 * length_a / 3], "boundary_cusp_z_m": [1.0e-4], "cusp_source": "test", "transit_time_s": 2.6e-6,
         "target_transits": 3.0, "extra_key_ignored": True},
    ]
    path = tmp_path / "runs.json"
    path.write_text(json.dumps({"schema": panel.SCHEMA, "runs": runs}), encoding="utf-8")
    return path


def _snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(folder.rglob("*")) if p.is_file()}


def test_interim_labels():
    assert panel.interim_title(0.3) == "INTERIM \u00b7 development \u00b7 t = 0.300 \u00b5s \u00b7 not a plateau"
    assert panel.short_label("divergent-exit-stack") == "ref" and panel.short_label("l1a-gs-v2-047-e3196a8aa5") == "047"
    assert panel.short_label("l1a-gs-v3-056-effcbc8686") == "056" and panel.short_label("weird") == "weird"
    assert panel.classify_boundary_cusps([7.3e-5, 0.0066, 0.0193, 0.0259], 0.0, 0.0259) == [7.3e-5, 0.0259]


def test_synthesised_summary_feeds_the_renderer_readers(tmp_path: Path):
    results = _synthetic_run(tmp_path, "d", truncated=False)
    protocol = json.loads((results / "protocol.json").read_text(encoding="utf-8"))
    summary = panel.synthesise_summary(protocol, frames_count=5, run_state={"finished": False}, protocol_sha256="ab" * 32)
    grid = video.grid_from_summary(summary)
    frames = load_frames(results)
    assert grid.node_shape == frames.maps["n_i_per_m3"].shape[1:] == (9, 25)
    assert video.run_constants(summary) == (W_MACRO, DT_STEP)
    assert summary["status"] == panel.INTERIM_STATUS and summary["synthesised"] is True
    assert "INTERIM" in summary["claim_boundary"] and "not a plateau" in summary["claim_boundary"].lower()
    assert summary["run_protocol_status"] == "synthetic_test_protocol" and summary["artifacts"]["frames"]["count"] == 5


def test_stage_mirror_skips_the_unloadable_frame_and_only_reads_the_results_dir(tmp_path: Path):
    results = _synthetic_run(tmp_path, "d", n_frames=5)
    before = _snapshot(results)
    mirror = tmp_path / "mirror"
    (mirror / "frames").mkdir(parents=True)
    (mirror / "frames" / "stale.npz").write_bytes(b"stale")
    report = panel.stage_mirror(results, mirror, log=lambda _: None)
    assert report["frames_staged"] == 5 and report["latest_end_step"] == 5 * FRAME_STEPS
    assert [s["file"] for s in report["frames_skipped"]] == ["frame-000005.npz"] and report["frames_dropped_after_gap"] == []
    assert report["link_mode"] and set(report["link_mode"]) <= {"symlink", "copy"}
    assert _snapshot(results) == before                                     # nothing written into the run's directory
    assert not (mirror / "frames" / "stale.npz").exists()                  # re-staging clears the previous mirror
    staged = load_frames(mirror)
    assert staged.count == 5 and staged.files == [f"frame-{i:06d}.npz" for i in range(5)]
    summary = json.loads((mirror / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == panel.INTERIM_STATUS and summary["run_state"]["frames_written"] == 5
    assert summary["protocol_sha256"] and len(summary["protocol_sha256"]) == 64
    # a gap ends the staged sequence: the frames after it are dropped, not stitched
    os.remove(results / "frames" / "frame-000002.npz")
    kept, skipped, dropped = panel.loadable_frames(results)
    assert [p.name for p, _, _ in kept] == ["frame-000000.npz", "frame-000001.npz"]
    assert dropped == ["frame-000003.npz", "frame-000004.npz"] and [s["file"] for s in skipped] == ["frame-000005.npz"]


def test_status_series_and_table_tolerate_a_partial_last_line(tmp_path: Path):
    results = _synthetic_run(tmp_path, "d", n_frames=3)
    series = panel.load_status_series(results)
    assert series["records"] == 3 * FRAME_STEPS // 200 and series["last"]["step"] == 3 * FRAME_STEPS
    assert np.all(np.isfinite(series["peak_cells_per_debye"])) and series["window_cells_per_debye"][-1] == pytest.approx(0.6)
    assert series["windowed_residual"][-1] == pytest.approx(-0.08)
    spec_ = panel.RunSpec(job_id="j", design_id="d", label="d", results=str(results), rho=1.0, transit_time_s=2.0e-6)
    row = panel.status_row(spec_, frames_staged=3, series=series)
    assert row["time_us"] == pytest.approx(3 * FRAME_STEPS * DT_STEP * 1e6) and row["transits"] == pytest.approx(3 * FRAME_STEPS * DT_STEP / 2.0e-6)
    assert row["discharge_ma"] == pytest.approx(1.2) and row["frames_written"] == 3 and row["frames_staged"] == 3 and row["triad_soft_ok"] is True
    table = panel.format_status_table([row])
    assert table.startswith("| design |") and "d d" in table and "(partial)" in table and "-8.0%" in table
    empty = panel.load_status_series(tmp_path / "nowhere")
    assert empty["records"] == 0 and empty["last"] is None and panel.status_row(spec_, series=empty)["time_us"] is None


def test_render_all_writes_panel_timeseries_banner_videos_and_validating_players(tmp_path: Path):
    a = _synthetic_run(tmp_path / "runs", "synthetic-cold", radial_cells=8, axial_cells=24, n_frames=5, seed=1)
    b = _synthetic_run(tmp_path / "runs", "synthetic-hot", radial_cells=6, axial_cells=30, n_frames=4, seed=2, density_scale=3.0)
    manifest = _manifest(tmp_path, a, b)
    out = tmp_path / "interim"
    report = panel.render_all(manifest, out, backend="pillow_gif", maps=("n_i_per_m3", video.IZ_KEY), fps=4, upscale=1, log=lambda _: None)
    # rows ordered by rho (the manifest listed them the other way round); the runs' directories were only read
    assert report["panel"]["rows"] == ["synthetic-cold", "synthetic-hot"] == report["timeseries"]["rows"]
    assert report["problems"] == [] and "not a plateau" in report["label"]
    assert not (a / "summary.json").exists() and not (b / "summary.json").exists()
    # the comparison figure: 2400 px wide, 2 rows x 3 columns, shared per-column scales declared with their basis
    with Image.open(report["panel"]["path"]) as im:
        assert im.size == (panel.FIGURE_WIDTH, report["panel"]["height"]) and im.size[1] > 400
    scales = report["panel"]["scales"]
    assert set(scales) == set(panel.PANEL_MAPS) and all("basis" in s for s in scales.values())
    assert scales["n_i_per_m3"]["kind"] == "log" and scales["n_i_per_m3"]["decades"] == pytest.approx(panel.NI_DECADES)
    assert scales["phi_v"]["kind"] == "signed" and scales["phi_v"]["lo"] < scales["phi_v"]["hi"] <= 300.0
    assert scales[video.IZ_KEY]["percentiles"] == [0.5, 99.5] and scales[video.IZ_KEY]["lo"] < scales[video.IZ_KEY]["hi"]
    assert len(report["panel"]["layout"]) == 6 and {row["map"] for row in report["panel"]["layout"]} == set(panel.PANEL_MAPS)
    # the same axial pixel scale for every row: 30 axial cells -> 31 nodes is the widest map, both rows share the factor
    widths = {row["design_id"]: row["w"] for row in report["panel"]["layout"]}
    assert widths["synthetic-hot"] / 31 == pytest.approx(widths["synthetic-cold"] / 25, rel=0.05)
    # the time-series strip
    with Image.open(report["timeseries"]["path"]) as im:
        assert im.size == (panel.FIGURE_WIDTH, 1000)
    assert [ax["key"] for ax in report["timeseries"]["axes"]] == ["discharge_a", "ionization_rate_per_s", "electrons", "peak_cells_per_debye"]
    assert report["timeseries"]["axes"][3]["hi"] >= panel.GATE_HARD
    # status table + report on disk
    status = (out / "interim-sweep-status.md").read_text(encoding="utf-8")
    assert status.startswith("# INTERIM") and "synthetic-cold" in status and "synthetic-hot" in status
    saved = json.loads((out / "interim-sweep-report.json").read_text(encoding="utf-8"))
    assert saved["panel"]["sha256"] == report["panel"]["sha256"] and len(saved["runs"]) == 2
    assert saved["runs"][0]["stage"]["frames_staged"] == 5 and saved["runs"][1]["stage"]["frames_staged"] == 4
    assert saved["runs"][0]["ionisation"]["window_frames"] >= 1 and saved["runs"][0]["status"]["frames_staged"] == 5
    # per-design videos wear the INTERIM banner (taller than the bare composed frame) and the players validate
    for design_id, frames_expected in (("synthetic-cold", 5), ("synthetic-hot", 4)):
        block = report["videos"][design_id]
        assert block["frames"] == frames_expected and set(block["videos"]) == {"n_i_per_m3", video.IZ_KEY} and block["backend"] == "pillow_gif"
        assert "INTERIM" in block["banner_example"] and "not a plateau" in block["banner_example"]
        for item in block["videos"].values():
            path = Path(item["path"])
            assert path.suffix == ".gif" and path.stat().st_size > 0 and path.parent == out / design_id / "video"
            with Image.open(path) as gif:
                assert gif.n_frames == frames_expected
        html = Path(block["html"]["path"]).read_text(encoding="utf-8")
        assert "INTERIM" in html.split("<title>")[1].split("</title>")[0]
        payload = json.loads(html.split('<script id="pic2d-data" type="application/json">')[1].split("</script>")[0])
        video.validate_player_payload(payload)
        assert payload["frame_count"] == frames_expected and "INTERIM DEVELOPMENT" in payload["claim_boundary"]
        assert payload["maps"][video.IZ_KEY]["window"]["causal"] is True
    assert report["videos"]["synthetic-cold"]["html"] and json.loads(Path(report["videos"]["synthetic-cold"]["html"]["path"]).read_text(encoding="utf-8")
                                                                    .split('type="application/json">')[1].split("</script>")[0])["cusp_z_m"] == pytest.approx([1.0e-4, 4.0e-4, 8.0e-4])


def test_banner_and_panel_are_deterministic(tmp_path: Path):
    image = np.full((20, 30, 3), 90, dtype=np.uint8)
    tall = panel.add_banner(image, "INTERIM banner")
    assert tall.shape == (50, 30, 3) and tuple(tall[45, 5]) == (90, 90, 90) and tuple(tall[2, 28]) == panel.BANNER_BG
    a = _synthetic_run(tmp_path / "runs", "synthetic-cold", n_frames=3, seed=3)
    b = _synthetic_run(tmp_path / "runs", "synthetic-hot", radial_cells=6, axial_cells=30, n_frames=3, seed=4)
    specs = panel.load_manifest(_manifest(tmp_path, a, b))
    assert [s.label for s in specs] == ["cold", "hot"] and not hasattr(specs[0], "extra_key_ignored")
    runs = [panel.prepare_run(s, tmp_path / "stage" / s.design_id, log=lambda _: None) for s in specs]
    first = panel.render_panel(runs, tmp_path / "p1.png", stamp="fixed")
    second = panel.render_panel(runs, tmp_path / "p2.png", stamp="fixed")
    assert first["sha256"] == second["sha256"]
    ts1 = panel.render_timeseries(runs, tmp_path / "t1.png", stamp="fixed")
    ts2 = panel.render_timeseries(runs, tmp_path / "t2.png", stamp="fixed")
    assert ts1["sha256"] == ts2["sha256"]
    # the CLI: status on the manifest, render without videos
    assert panel.main(["status", "--manifest", str(tmp_path / "runs.json"), "--json"]) == 0
    assert panel.main(["render", "--manifest", str(tmp_path / "runs.json"), "--out-dir", str(tmp_path / "cli"), "--no-videos"]) == 0
    assert (tmp_path / "cli" / "interim-sweep-panel.png").is_file() and (tmp_path / "cli" / "interim-sweep-timeseries.png").is_file()
