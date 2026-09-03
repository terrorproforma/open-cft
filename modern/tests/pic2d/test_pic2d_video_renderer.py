"""Time-series renderer: frame -> palette image mapping, fixed scales, writers, HTML player payload, determinism."""

from __future__ import annotations

import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from cft_revival.pic2d.fields import uniform_field_map
from cft_revival.pic2d.frames import SCALAR_KEYS, FrameSet, load_frames
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


# -- ionisation panel: event counts, causal window, resolution mask, robust scale ---------------------------

W_MACRO = 6.0e4
DT_STEP = 1.5e-12
FRAME_STEPS = 20000                       # 30 ns frames, as recorded by the plume runs
IZ = video.IZ_KEY


class _Masks:
    """Stand-in for MeshMasks: plasma_node + shape_volume_m3 only (what the renderer reads)."""

    def __init__(self, plasma: np.ndarray, volume: np.ndarray) -> None:
        self.plasma_node = plasma
        self.shape_volume_m3 = volume


def _poisson_frames(true_rate: np.ndarray, volume: np.ndarray, plasma: np.ndarray, n: int, *, seed: int = 11,
                    frame_steps: int = FRAME_STEPS) -> tuple[FrameSet, np.ndarray]:
    """Frames whose ionisation map is exactly what the recorder writes for Poisson macro-events of the true rate.

    events ~ Poisson(true_rate V dt / W) per node per frame; map = events W / (V dt) (float32, as stored).
    Returns the frame set and the integer event counts."""

    rng = np.random.default_rng(seed)
    dt_frame = frame_steps * DT_STEP
    expected = true_rate * volume * dt_frame / W_MACRO
    counts = rng.poisson(np.broadcast_to(expected, (n, *true_rate.shape))).astype(np.float64)
    counts[:, ~plasma] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(plasma[None], counts * W_MACRO / (volume[None] * dt_frame), 0.0)
    shape = true_rate.shape
    maps = {key: np.full((n, *shape), 1e17, dtype=np.float32) for key in video.DEFAULT_MAPS}
    maps[IZ] = rate.astype(np.float32)
    maps["sample_count_e"] = np.full((n, *shape), 50.0, dtype=np.float32)
    maps["phi_v"] = np.full((n, *shape), 100.0, dtype=np.float32)
    maps["t_e_ev"] = np.full((n, *shape), 5.0, dtype=np.float32)
    scalars = {key: np.linspace(1.0, 2.0, n) for key in SCALAR_KEYS}
    start = np.arange(n, dtype=np.int64) * frame_steps
    frames = FrameSet("s", "float32", start, start + frame_steps, (start + frame_steps) * DT_STEP, maps, {},
                      np.zeros((n, *shape), dtype=np.float32), scalars, [f"f{i}" for i in range(n)])
    return frames, counts


def _channel_like() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8 radial x 12 axial nodes: cylindrical shape volumes (axis node tiny), one ionisation band, a wall row outside."""

    nr, nz = 8, 12
    dr = dz = 5.0e-5
    r = np.arange(nr) * dr
    volume = np.where(r > 0, 2.0 * np.pi * r * dr * dz, np.pi * (dr / 2.0) ** 2 * dz)[:, None] * np.ones((1, nz))
    plasma = np.ones((nr, nz), dtype=bool)
    plasma[-1, :] = False                                   # wall row
    true_rate = np.zeros((nr, nz))
    true_rate[1:6, :] = 4.0e24                              # ionising band, interior nodes (1.6-7.9 expected events per 30 ns frame)
    true_rate[3, 4:8] = 8.0e24                              # a brighter core
    true_rate[0, :] = 4.0e24                                # the axis ionises at the band rate too (tiny volume -> 0.2 events per frame)
    return true_rate, volume, plasma


def test_ionisation_events_reproduce_the_recorded_counts_and_the_domain_integral():
    true_rate, volume, plasma = _channel_like()
    frames, counts = _poisson_frames(true_rate, volume, plasma, n=40)
    events = video.ionisation_events(frames, volume, plasma, W_MACRO, DT_STEP)
    # rate x V x dt / W recovers the deposited counts (float32 storage of the rate -> ~1e-7 relative)
    np.testing.assert_allclose(events, counts, rtol=1e-5, atol=1e-4)
    assert np.all(events[:, ~plasma] == 0.0)
    # the domain total per frame is the integer number of macro-events of the interval
    totals = events.sum(axis=(1, 2))
    np.testing.assert_allclose(totals, np.rint(totals), atol=1e-3)
    # S of a frame = sum(rate V) = total events W / dt_frame
    s_frame = (frames.maps[IZ].astype(np.float64) * volume[None]).sum(axis=(1, 2))
    np.testing.assert_allclose(s_frame, totals * W_MACRO / (FRAME_STEPS * DT_STEP), rtol=1e-5)


def test_causal_window_sum_is_trailing_and_partial_at_the_start():
    values = np.arange(1.0, 8.0)[:, None] * np.ones((1, 2))
    out = video.causal_window_sum(values, 3)
    np.testing.assert_allclose(out[:, 0], [1, 3, 6, 9, 12, 15, 18])       # 1 | 1+2 | 1+2+3 | 2+3+4 | ...
    np.testing.assert_allclose(video.causal_window_sum(values, 1), values)
    np.testing.assert_allclose(video.causal_window_sum(values, 100)[:, 1], np.cumsum(values[:, 1]))
    with pytest.raises(ValueError):
        video.causal_window_sum(values, 0)


def test_windowed_rate_converges_to_the_true_rate_and_is_causal():
    true_rate, volume, plasma = _channel_like()
    n = 200
    frames, _ = _poisson_frames(true_rate, volume, plasma, n=n)
    band = plasma & (true_rate > 0) & (np.arange(volume.shape[0])[:, None] > 0)           # interior ionising nodes
    expected_per_frame = true_rate * volume * FRAME_STEPS * DT_STEP / W_MACRO
    # a single frame: O(1) relative error (mean expected count per node is small)
    one = video.windowed_ionisation(frames, volume, plasma, W_MACRO, DT_STEP, 1)
    err1 = np.sqrt(np.mean(((one["rate"][-1] - true_rate)[band] / true_rate[band]) ** 2))
    # the full window: relative error ~ 1/sqrt(N_events); every band node within 5 sigma, RMS shrinks ~ sqrt(K)
    full = video.windowed_ionisation(frames, volume, plasma, W_MACRO, DT_STEP, n)
    assert full["frames_in_window"][-1] == n and full["window_s"][-1] == pytest.approx(n * FRAME_STEPS * DT_STEP)
    n_expected = expected_per_frame * n
    sigma = true_rate / np.sqrt(np.where(n_expected > 0, n_expected, 1.0))
    z = ((full["rate"][-1] - true_rate) / np.where(sigma > 0, sigma, 1.0))[band]
    assert np.all(np.abs(z) < 5.0)
    errn = np.sqrt(np.mean(((full["rate"][-1] - true_rate)[band] / true_rate[band]) ** 2))
    assert errn < err1 / (np.sqrt(n) / 3.0)
    assert abs(np.mean(full["rate"][-1][band] / true_rate[band]) - 1.0) < 0.05
    # windowed rate == time-weighted mean of the frame rates (equal intervals here -> plain mean)
    k = 12
    win = video.windowed_ionisation(frames, volume, plasma, W_MACRO, DT_STEP, k)
    np.testing.assert_allclose(win["rate"][k - 1:], np.array([frames.maps[IZ][i - k + 1:i + 1].astype(np.float64).mean(axis=0) for i in range(k - 1, n)]),
                               rtol=1e-5, atol=1e12)
    np.testing.assert_allclose(win["frames_in_window"][:k], np.arange(1, k + 1))
    # causal: perturbing later frames leaves earlier windowed values untouched
    perturbed = deepcopy(frames)
    perturbed.maps[IZ][60:] *= 10.0
    again = video.windowed_ionisation(perturbed, volume, plasma, W_MACRO, DT_STEP, k)
    np.testing.assert_array_equal(again["rate"][:60], win["rate"][:60])
    assert not np.array_equal(again["rate"][60], win["rate"][60])


def test_windowed_rate_integrates_to_the_window_mean_ionisation_rate():
    true_rate, volume, plasma = _channel_like()
    frames, counts = _poisson_frames(true_rate, volume, plasma, n=30)
    k = 7
    win = video.windowed_ionisation(frames, volume, plasma, W_MACRO, DT_STEP, k)
    integral = (win["rate"] * volume[None]).sum(axis=(1, 2))
    s_frame = (frames.maps[IZ].astype(np.float64) * volume[None]).sum(axis=(1, 2))
    dt = video.frame_interval_s(frames, DT_STEP)
    for i in range(frames.count):
        lo = max(0, i - k + 1)
        s_mean = float((s_frame[lo:i + 1] * dt[lo:i + 1]).sum() / dt[lo:i + 1].sum())
        assert integral[i] == pytest.approx(s_mean, rel=1e-5)
        assert integral[i] == pytest.approx(counts[lo:i + 1].sum() * W_MACRO / dt[lo:i + 1].sum(), rel=1e-5)


def test_single_event_axis_node_is_unresolved_not_the_colour_top():
    true_rate, volume, plasma = _channel_like()
    n = 40
    frames, _ = _poisson_frames(true_rate, volume, plasma, n=n, seed=5)
    # force exactly one event on an axis node in one frame and none elsewhere on the axis
    frames.maps[IZ][:, 0, :] = 0.0
    dt_frame = FRAME_STEPS * DT_STEP
    frames.maps[IZ][20, 0, 5] = W_MACRO / (volume[0, 5] * dt_frame)
    one_event_rate = float(frames.maps[IZ][20, 0, 5])
    # one event on the small-volume axis node is ~2.5x the core's TRUE rate; the pre-v0.2 renderer (per-frame map, scale =
    # run maximum over electron-resolved nodes) painted that node at the top of the colour bar
    assert one_event_rate > 2.0 * true_rate.max()
    old_scale = video.colour_scale(frames, IZ, plasma, 20)
    assert old_scale["hi"] >= one_event_rate
    old_idx = video.index_frame(frames.maps[IZ][20], frames.maps["sample_count_e"][20], plasma, old_scale, 20)
    assert old_idx[-1, 5] >= 240
    masks = _Masks(plasma, volume)
    iz = video.prepare_ionisation(frames, masks, W_MACRO, DT_STEP, window=12, min_events=20)
    assert iz["scale"]["hi"] < 0.75 * one_event_rate                       # a robust percentile of the resolved windowed nodes, not that node
    assert iz["scale"]["hi"] < 1.3 * true_rate.max()                       # ... i.e. the core's true rate, up to Poisson noise
    idx = video.index_frame(iz["rate"][20], iz["events"][20], plasma, iz["scale"], iz["min_events"])
    assert idx[-1, 5] == video.MASK_INDEX                                  # image row -1 = radial row 0 (the axis): grey, not yellow
    assert idx[-1, 5] != 253
    # the well-sampled band rows (radial 2..5, >= 37 expected windowed events) are coloured; the wall row is body
    assert idx[0, :].tolist() == [video.BODY_INDEX] * plasma.shape[1]
    band_rows = idx[-6:-2, :]
    assert (band_rows != video.MASK_INDEX).sum() > 0.8 * band_rows.size
    # the mask is on windowed EVENTS (dashboard 6bd5e5b0 semantics): a node with >= 20 windowed events is resolved
    resolved = iz["events"][20] >= 20
    assert np.array_equal(iz["resolved"][20], resolved & plasma)


def test_choose_window_is_the_smallest_k_meeting_the_median_target():
    true_rate, volume, plasma = _channel_like()
    frames, _ = _poisson_frames(true_rate, volume, plasma, n=60, seed=2)
    events = video.ionisation_events(frames, volume, plasma, W_MACRO, DT_STEP)
    samples = frames.maps["sample_count_e"]
    k = video.choose_window(events, samples, plasma, target_median_events=10.0, min_samples=20)
    assert 1 <= k <= 60

    def median_for(kk: int) -> float:
        ev = events[:, plasma]
        es = np.asarray(samples, dtype=np.float64)[:, plasma]
        sums = np.array([ev[i - kk + 1:i + 1].sum(axis=0) for i in range(kk - 1, 60)])
        smp = np.array([es[i - kk + 1:i + 1].sum(axis=0) for i in range(kk - 1, 60)])
        pick = sums[(smp >= 20) & (sums >= 1.0)]
        return float(np.median(pick)) if pick.size else 0.0

    assert median_for(k) >= 10.0
    assert all(median_for(kk) < 10.0 for kk in range(1, k))
    # the expected count per band node per frame is ~1.6-7.9, so the window is a few frames, not the run
    assert 2 <= k <= 30
    # unreachable target -> the whole run (capped by max_window)
    assert video.choose_window(events, samples, plasma, target_median_events=1e9) == 60
    assert video.choose_window(events, samples, plasma, target_median_events=1e9, max_window=7) == 7
    # under-sampled nodes do not count as resolved for the selection
    assert video.choose_window(events, np.zeros_like(samples), plasma, target_median_events=10.0) == 60


def test_ionisation_scale_is_a_robust_percentile_range_of_the_resolved_windowed_values():
    rate = np.zeros((3, 2, 4))
    events = np.zeros((3, 2, 4))
    plasma = np.ones((2, 4), dtype=bool)
    rate[:, 0, :] = np.array([1e22, 2e22, 4e22, 8e22])
    events[:, 0, :] = 25.0
    rate[:, 1, :] = 1e26                                        # unresolved outliers (1 event weight each)
    events[:, 1, :] = 1.0
    scale = video.ionisation_scale(rate, events, plasma, 20.0, (0.0, 100.0))
    assert scale["kind"] == "log" and scale["lo"] == pytest.approx(1e22) and scale["hi"] == pytest.approx(8e22)
    assert scale["decades"] == pytest.approx(np.log10(8.0)) and scale["percentiles"] == [0.0, 100.0]
    robust = video.ionisation_scale(rate, events, plasma, 20.0, (25.0, 75.0))
    assert 1e22 < robust["lo"] < robust["hi"] < 8e22
    # degenerate (one value) -> a decade either side; nothing resolved -> fall back to every positive value
    flat = video.ionisation_scale(np.full((1, 1, 1), 3e22), np.full((1, 1, 1), 30.0), np.ones((1, 1), bool), 20.0)
    assert flat["lo"] == pytest.approx(3e21) and flat["hi"] == pytest.approx(3e23)
    none = video.ionisation_scale(rate, np.zeros_like(events), plasma, 20.0, (0.0, 100.0))
    assert none["lo"] == pytest.approx(1e22) and none["hi"] == pytest.approx(1e26)


def test_ionisation_legend_and_payload_block_declare_window_mask_and_scale():
    true_rate, volume, plasma = _channel_like()
    frames, _ = _poisson_frames(true_rate, volume, plasma, n=16, seed=9)
    iz = video.prepare_ionisation(frames, _Masks(plasma, volume), W_MACRO, DT_STEP, window=6, min_events=20)
    assert iz["window"] == 6 and iz["auto"] is False and iz["nominal_window_s"] == pytest.approx(6 * FRAME_STEPS * DT_STEP)
    title, legend = video.ionisation_legend(iz, 2)
    assert title == "window 3/6 frames = 90 ns (causal, partial)"
    title_full, legend_full = video.ionisation_legend(iz, 15)
    assert title_full == "window 6/6 frames = 180 ns (causal)"
    assert "log10" in legend_full[0] and "0.5-99.5 pct" in legend_full[0] and "fixed over the run" in legend_full[0]
    assert legend_full[1].startswith("grey = unresolved: < 20 macro-ionisation events") and "no spatial smoothing" in legend_full[1]
    assert legend_full[2].startswith(f"resolved: {int(iz['resolved_nodes'][15])} of {int(plasma.sum())} plasma nodes carry")
    assert all(line.isascii() for line in legend + legend_full + [title])
    block = video.ionisation_payload_block(iz)
    assert block["window"]["frames"] == 6 and block["window"]["causal"] is True and block["window"]["frames_in_window"] == [min(i + 1, 6) for i in range(16)]
    assert block["mask"]["min_count"] == 20.0 and "fewer than 20 macro-ionisation events" in block["mask"]["note"]
    assert "not the per-frame maximum" in block["scale_note"] and "99.5th percentile" in block["scale_note"]
    # the composed frame grows by one 16 px text row per extra legend line and carries the same map
    scale = iz["scale"]
    idx = video.index_frame(iz["rate"][15], iz["events"][15], plasma, scale, iz["min_events"])
    default = video.compose_video_frame(idx, scale, IZ, 15, frames, _grid_stub(plasma.shape), upscale=2, min_samples=20)
    tall = video.compose_video_frame(idx, scale, IZ, 15, frames, _grid_stub(plasma.shape), upscale=2, min_samples=20, title_suffix=title_full, legend=legend_full)
    assert tall.shape[0] == default.shape[0] + 32 and tall.shape[1] == default.shape[1]
    auto = video.prepare_ionisation(frames, _Masks(plasma, volume), W_MACRO, DT_STEP, min_events=20)
    assert auto["auto"] is True and auto["window"] == video.choose_window(auto["frame_events"], frames.maps["sample_count_e"], plasma)


def _grid_stub(shape: tuple[int, int]):
    """A straight 50 um-cell channel whose node shape matches the synthetic maps (outline decoration only)."""

    from cft_revival.pic2d.models import ChannelGeometry, Grid2D
    nr, nz = shape
    radius, length = (nr - 1) * 5.0e-5, (nz - 1) * 5.0e-5
    return Grid2D(ChannelGeometry(radius, 0.0, length, length, radius), nr - 1, nz - 1)


def test_player_payload_validation_requires_the_ionisation_window_declaration():
    true_rate, volume, plasma = _channel_like()
    frames, _ = _poisson_frames(true_rate, volume, plasma, n=5, seed=1)
    grid = _grid_stub(plasma.shape)
    iz = video.prepare_ionisation(frames, _Masks(plasma, volume), W_MACRO, DT_STEP, window=3, min_events=20)
    summary = {"experiment_id": "x", "model_version": "v", "status": "development_screening_not_preregistered", "claim_boundary": "cb",
               "provenance": {"config": {"macro_weight": W_MACRO, "dt_s": DT_STEP}}, "artifacts": {"frames": {"sha256": "ab"}}}
    payload = video.build_player_payload(Path("."), frames, summary, grid, maps=(IZ, "n_e_per_m3"), factor=1, ionisation=iz)
    video.validate_player_payload(payload)
    assert payload["maps"][IZ]["window"]["frames"] == 3 and payload["maps"][IZ]["mask"]["min_count"] == 20.0
    assert payload["maps"]["n_e_per_m3"]["mask"] == {"counts": "electron samples", "min_count": 20.0, "note": "grey: fewer than 20 electron samples in the frame interval"}
    assert "shot-noise" in payload["claim_statement"] and "causal rolling window" in payload["claim_statement"]
    # the payload built without a prepared block computes it from the summary constants
    same = video.build_player_payload(Path("."), frames, summary, grid, maps=(IZ,), factor=1)
    assert same["maps"][IZ]["window"]["auto"] is True and same["maps"][IZ]["png"]
    for mutate in (
        lambda p: p["maps"][IZ].pop("window"),
        lambda p: p["maps"][IZ]["window"].update(causal=False),
        lambda p: p["maps"][IZ]["window"]["frames_in_window"].__setitem__(0, 3),
        lambda p: p["maps"][IZ].pop("mask"),
        lambda p: p["maps"][IZ]["scale"].pop("percentiles"),
        lambda p: p["maps"]["n_e_per_m3"]["mask"].update(min_count=-1),
    ):
        bad = deepcopy(payload)
        mutate(bad)
        with pytest.raises(ValueError):
            video.validate_player_payload(bad)


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
    import base64
    import io

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
    # the ionisation panel on a real (tiny) run: window declared, suffix applied, report carries the choices
    iz_report = video.render_run(results, tmp_path / "iz", maps=(video.IZ_KEY,), fps=4, backend="pillow_gif", cusp_z_m=[], upscale=2, suffix="-v2",
                                 iz_window=2, iz_min_events=1.0)
    assert Path(iz_report["videos"][video.IZ_KEY]["path"]).name == "pic2d-run-ionization_rate_per_m3_s-v2.gif"
    assert Path(iz_report["html"]["path"]).name == "pic2d-run-timeseries-v2.html"
    info = iz_report["ionisation"]
    assert info["window_frames"] == 2 and info["auto"] is False and info["min_events"] == 1.0 and info["window_s"] == pytest.approx(2 * 40 * 5e-12)
    assert info["macro_weight"] == 6.0e5 and info["dt_s"] == 5.0e-12 and info["scale"]["kind"] == "log" and info["scale"]["percentiles"] == [0.5, 99.5]
    html = Path(iz_report["html"]["path"]).read_text(encoding="utf-8")
    payload = json.loads(html.split('<script id="pic2d-data" type="application/json">')[1].split("</script>")[0])
    video.validate_player_payload(payload)
    assert payload["maps"][video.IZ_KEY]["window"]["frames_in_window"] == [1, 2, 2, 2]
    auto = video.main([str(results), "--out", str(tmp_path / "cli2"), "--no-video", "--maps", video.IZ_KEY, "--suffix=-auto", "--iz-min-events", "1"])
    assert auto == 0 and (tmp_path / "cli2" / "pic2d-run-timeseries-auto.html").is_file()
