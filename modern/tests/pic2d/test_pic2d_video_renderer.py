"""Time-series renderer: frame -> palette image mapping, fixed scales, writers, HTML player payload, determinism."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from cft_revival.pic2d.fields import uniform_field_map
from cft_revival.pic2d.frames import FrameSet, SCALAR_KEYS, load_frames
from cft_revival.pic2d.mcc import XenonCrossSections
from experiments.pic2d_cft_steady_state_v1 import run as runner

MODERN = Path(__file__).resolve().parents[2]
PLUME_PROTOCOL = MODERN / "experiments" / "pic2d_cft_plume_v1" / "protocol.json"

spec = importlib.util.spec_from_file_location("render_pic2d_video", MODERN / "visualization" / "render_pic2d_video.py")
video = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(video)


def _synthetic_frames(n: int = 3, shape: tuple[int, int] = (6, 10)) -> FrameSet:
    rng = np.random.default_rng(3)
    maps = {key: rng.random((n, *shape)).astype(np.float32) * 1e17 for key in video.DEFAULT_MAPS}
    maps["sample_count_e"] = np.full((n, *shape), 50.0, dtype=np.float32)
    maps["phi_v"] = (rng.random((n, *shape)) * 300.0).astype(np.float32)
    maps["t_e_ev"] = (rng.random((n, *shape)) * 10.0).astype(np.float32)
    scalars = {key: np.linspace(1.0, 2.0, n) for key in SCALAR_KEYS}
    return FrameSet("s", "float32", np.arange(n) * 10, np.arange(1, n + 1) * 10, np.arange(1, n + 1) * 1e-9, maps, {},
                    np.zeros((n, *shape), dtype=np.float32), scalars, [f"f{i}" for i in range(n)])


def test_index_frame_maps_values_mask_and_body_to_declared_palette_entries():
    values = np.array([[1e12, 1e14, 1e16], [0.0, np.nan, 1e20]])
    counts = np.array([[50, 50, 50], [50, 50, 5]])
    plasma = np.array([[True, True, True], [True, True, False]])
    scale = {"kind": "log", "lo": 1e12, "hi": 1e16, "decades": 4.0}
    idx = video.index_frame(values, counts, plasma, scale, 20)
    assert idx.dtype == np.uint8 and idx.shape == (2, 3)
    # row 0 of the image is the LAST radial row (r increases upward)
    assert idx[1, 0] == 0 and idx[1, 1] == 126 and idx[1, 2] == 253          # lo -> 0, mid-decade -> rint(126.5) = 126, hi -> 253
    assert idx[0, 0] == video.MASK_INDEX and idx[0, 1] == video.MASK_INDEX   # zero / NaN on a log scale are undefined -> grey
    assert idx[0, 2] == video.BODY_INDEX                                     # outside the plasma mask wins over the sample mask
    linear = video.index_frame(np.array([[0.0, 5.0, 10.0, 20.0]]), np.full((1, 4), 50), np.ones((1, 4), bool), {"kind": "linear", "lo": 0.0, "hi": 10.0}, 20)
    np.testing.assert_array_equal(linear[0], [0, 126, 253, 253])              # clipped at the fixed maximum
    under = video.index_frame(np.array([[5.0]]), np.array([[19]]), np.ones((1, 1), bool), {"kind": "linear", "lo": 0.0, "hi": 10.0}, 20)
    assert under[0, 0] == video.MASK_INDEX
    rgb = video.to_rgb(idx, "log")
    assert rgb.shape == (2, 3, 3) and tuple(rgb[0, 2]) == video.BODY_RGB and tuple(rgb[0, 0]) == video.MASK_RGB
    assert tuple(rgb[1, 0]) == (68, 1, 84) and tuple(rgb[1, 2]) == (253, 231, 37)   # viridis ends
    assert video.palette("signed").shape == (256, 3) and tuple(video.palette("signed")[0]) == (59, 76, 192)


def test_fixed_colour_scale_downsample_and_outline():
    frames = _synthetic_frames()
    plasma = np.ones((6, 10), bool)
    plasma[4:, :5] = False
    scale = video.colour_scale(frames, "n_e_per_m3", plasma, 20)
    assert scale["kind"] == "log" and scale["hi"] == pytest.approx(float(frames.maps["n_e_per_m3"][:, plasma].max()))
    assert scale["lo"] == pytest.approx(scale["hi"] / 1e4)
    phi = video.colour_scale(frames, "phi_v", plasma, 20)
    assert phi["kind"] == "signed" and phi["lo"] < phi["hi"] and phi["lo"] == pytest.approx(float(frames.maps["phi_v"][:, plasma].min()))
    # the mask threshold applies to the scale too: heavily under-sampled frames do not set the maximum
    frames.maps["sample_count_e"][0] = 0.0
    frames.maps["n_e_per_m3"][0] = 1e30
    assert video.colour_scale(frames, "n_e_per_m3", plasma, 20)["hi"] == pytest.approx(scale["hi"])
    block = video.downsample(np.arange(24, dtype=float).reshape(4, 6), 2)
    np.testing.assert_allclose(block, [[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]])
    np.testing.assert_allclose(video.downsample(np.ones((5, 5)), 2, how="sum"), np.full((2, 2), 4.0))   # trailing partial blocks dropped
    outline = video.mask_outline(np.array([[True, True], [True, False]]))
    # image row 0 = last mask row: the single non-plasma node sits at image (col 1, row 0); its two inner edges
    assert sorted(outline) == sorted([[[1, 0], [1, 1]], [[1, 1], [2, 1]]])


def test_writers_produce_files(tmp_path: Path):
    images = [np.full((9, 11, 3), c, dtype=np.uint8) for c in (10, 120, 250)]
    path, backend = video.write_video(images, tmp_path / "clip", fps=5, backend="pillow_gif")
    assert path.suffix == ".gif" and backend == "pillow_gif" and path.stat().st_size > 0
    from PIL import Image
    with Image.open(path) as gif:
        assert gif.n_frames == 3
    if shutil.which("ffmpeg"):
        mp4, used = video.write_video(images, tmp_path / "clip2", fps=5, backend="ffmpeg")
        assert mp4.suffix == ".mp4" and used == "ffmpeg" and mp4.stat().st_size > 0   # odd sizes are padded to even
    assert video.available_backend() in ("imageio_ffmpeg", "ffmpeg", "pillow_gif")


def _tiny_run(tmp_path: Path) -> tuple[Path, Path]:
    protocol = deepcopy(runner.load_protocol(PLUME_PROTOCOL))
    protocol["geometry"]["body_dielectric_radius_m"] = 0.0045
    protocol["case"].update({"radial_cells": 48, "axial_cells": 144, "macro_weight": 6.0e5})
    protocol["numerics"].update({"dt_s": 5.0e-12, "device_sync_steps": 20, "series_interval_steps": 20, "checkpoint_every_steps": 80,
                                 "averaging_window_steps": 160, "frame_recorder": {"cadence_steps": 40, "precision": "float32"}})
    protocol["numerics"]["stability_reference"]["density_per_m3"] = 1.0e16
    protocol["operating_point"]["seed_plasma_density_per_m3"] = 5.0e15
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    config = runner.build_config(protocol, backend="cpu")
    results = tmp_path / "run" / "results"
    runner.run_steady_state(protocol, results, backend="cpu", field_map=uniform_field_map(config.grid, 0.02), cross_sections=XenonCrossSections.from_file(),
                            max_steps=160, protocol_path=protocol_path, log=lambda _: None)
    return results, protocol_path


def test_render_run_is_deterministic_and_the_player_payload_validates(tmp_path: Path):
    results, protocol_path = _tiny_run(tmp_path)
    out = tmp_path / "video"
    report = video.render_run(results, out, maps=("n_e_per_m3", "phi_v"), fps=4, backend="pillow_gif", cusp_z_m=[0.006, 0.012], upscale=2)
    assert report["frames"] == 4 and report["backend"] == "pillow_gif" and set(report["videos"]) == {"n_e_per_m3", "phi_v"}
    assert Path(report["videos"]["n_e_per_m3"]["path"]).suffix == ".gif" and Path(report["html"]["path"]).name == "pic2d-run-timeseries.html"
    html = Path(report["html"]["path"]).read_text(encoding="utf-8")
    payload = json.loads(html.split('<script id="pic2d-data" type="application/json">')[1].split("</script>")[0])
    video.validate_player_payload(payload)
    assert payload["frame_count"] == 4 and payload["cadence_steps"] == 40 and payload["interval_s"] == pytest.approx(40 * 5e-12)
    assert payload["downsample_factor"] == 2 and payload["image_shape"] == [24, 72] and payload["cusp_z_m"] == [0.006, 0.012]
    assert payload["min_samples"] == video.MIN_SAMPLES_DEFAULT and "fewer than 20" in payload["min_samples_note"]
    assert set(payload["maps"]) == {"n_e_per_m3", "phi_v"} and all(len(m["png"]) == 4 for m in payload["maps"].values())
    assert payload["maps"]["n_e_per_m3"]["scale"]["kind"] == "log" and payload["maps"]["phi_v"]["scale"]["kind"] == "signed"
    assert set(payload["series"]) == {k for k, _, _ in video.SERIES} and payload["series"]["electrons"]["values"][0] > 0
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    assert payload["frames_sha256"] == summary["artifacts"]["frames"]["sha256"] and payload["protocol_sha256"] == summary["protocol_sha256"]
    assert payload["domain"]["has_plume"] is True and payload["body_outline"]
    # the embedded PNGs decode to the declared shape and use only palette indices
    import base64, io
    from PIL import Image
    with Image.open(io.BytesIO(base64.b64decode(payload["maps"]["n_e_per_m3"]["png"][0]))) as im:
        assert im.mode == "P" and im.size == (72, 24)
    # claim boundary phrases are enforced
    bad = deepcopy(payload)
    bad["claim_statement"] = "great thruster"
    with pytest.raises(ValueError):
        video.validate_player_payload(bad)
    bad = deepcopy(payload)
    bad["maps"]["phi_v"]["png"].pop()
    with pytest.raises(ValueError):
        video.validate_player_payload(bad)
    # deterministic: rendering again gives byte-identical HTML and GIF
    again = video.render_run(results, tmp_path / "video2", maps=("n_e_per_m3", "phi_v"), fps=4, backend="pillow_gif", cusp_z_m=[0.006, 0.012], upscale=2)
    assert again["html"]["sha256"] == report["html"]["sha256"]
    assert again["videos"]["n_e_per_m3"]["sha256"] == report["videos"]["n_e_per_m3"]["sha256"]
    # the composed video frame carries the map, the colour bar and the series strip
    frames = load_frames(results)
    grid = video.grid_from_summary(summary)
    scale = video.colour_scale(frames, "n_e_per_m3", video._plasma(grid), 20)
    idx = video.index_frame(frames.maps["n_e_per_m3"][0], frames.maps["sample_count_e"][0], video._plasma(grid), scale, 20)
    image = video.compose_video_frame(idx, scale, "n_e_per_m3", 0, frames, grid, upscale=2, min_samples=20, cusp_z_m=[0.006])
    assert image.dtype == np.uint8 and image.shape[2] == 3 and image.shape[1] == idx.shape[1] * 2 + 90
    # the CLI entry point renders the HTML only
    assert video.main([str(results), "--out", str(tmp_path / "cli"), "--no-video", "--maps", "t_e_ev", "--protocol", str(protocol_path)]) == 0
    assert (tmp_path / "cli" / "pic2d-run-timeseries.html").is_file()
