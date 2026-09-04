"""Render a PIC-2D run's recorded frames (frames/frame-NNNNNN.npz) as a video and an offline HTML player.

Inputs: a results directory of the steady-state runner with the frame recorder on (v2.0):
``frames/frame-NNNNNN.npz`` (exact interval averages of the node maps), ``summary.json``
(grid, protocol hash, frames manifest), optionally the cusp planes.

Outputs (deterministic for the same frames):
  <out>/pic2d-<run>-<map>.mp4 | .gif   one video per requested map, fixed colour scale across
                                       frames (log with a floor for the densities, robust log
                                       percentiles for the windowed ionisation rate, symmetric /
                                       linear for phi and T_e), grey mask for cells sampled by fewer
                                       than N macro-particles (N macro-ionisation events for the
                                       ionisation panel), thruster body dark, time stamp and the
                                       scalar diagnostics printed per frame, and a synchronised
                                       time-series strip (I_d, I_beam, N_e, n_g, T)
  <out>/pic2d-<run>-timeseries.html    offline player: frames embedded as 2x-downsampled palette
                                       PNGs (the same fixed scale), scrubber / play / speed, map
                                       selector, body outline + cusp planes drawn, time series under
                                       the map with a cursor, keyboard arrows

Video backend, in the declared order: ``imageio_ffmpeg`` (if importable), the ``ffmpeg``
executable on PATH (raw RGB pipe -> H.264 yuv420p), else an animated GIF via Pillow.

Ionisation-rate panel (v0.2). A single frame's ionisation map is shot-noise dominated by
construction: the map is ``events * W / (V_node * dt_frame)`` where ``events`` is the
bilinear-deposited weight of the macro-ionisation events of the interval (attempt 6: 30 ns
frames, ~73 % of the electron-resolved nodes hold zero events, one event on an axis node is
~3e25 m^-3 s^-1 and sets a per-frame maximum).  The panel therefore shows a CAUSAL rolling
window of K frames: windowed events = trailing sum of the per-frame events (partial window
over the first K-1 frames), windowed rate = windowed events * W / (V_node * windowed time),
which is the exact time-weighted mean of the frame rates and integrates to the mean
ionisation rate S of the window.  Nodes with fewer than N windowed events are grey
("unresolved", same semantics as the dashboards' ionisation-event mask, default N = 20 =
``MIN_SAMPLES_DEFAULT``), never zero.  K defaults to the smallest window for which the median
electron-resolved node carrying at least one event weight holds >= 10 events
(``choose_window``).  The colour scale is log10 between the 0.5th and 99.5th percentile of the
resolved windowed values over the whole run (not the per-frame maximum) and the window, the
mask and the scale are written into the panel legend.  No spatial smoothing is applied.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

MODERN = Path(__file__).resolve().parents[1]
if str(MODERN / "src") not in sys.path:
    sys.path.insert(0, str(MODERN / "src"))
if str(MODERN) not in sys.path:
    sys.path.insert(0, str(MODERN))

from cft_revival.pic2d.frames import FrameSet, load_frames
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import ChannelGeometry, Grid2D

SCHEMA = "cft-pic2d-timeseries-player/0.2.0"
MIN_SAMPLES_DEFAULT = 20
LOG_DECADES = 4.0
IZ_KEY = "ionization_rate_per_m3_s"
IZ_TARGET_MEDIAN_EVENTS = 10.0          # window selection: median resolved event-bearing node holds >= this many events
IZ_PERCENTILES = (0.5, 99.5)            # fixed colour range of the windowed rate over the run (robust, not the max)
DEFAULT_MAPS = ("n_e_per_m3", "n_i_per_m3", "phi_v", "t_e_ev", "ionization_rate_per_m3_s")
MAP_LABELS = {
    "n_e_per_m3": ("Electron density n_e", "m^-3", "log"),
    "n_i_per_m3": ("Ion density n_i", "m^-3", "log"),
    "phi_v": ("Potential phi", "V", "signed"),
    "t_e_ev": ("Electron temperature T_e", "eV", "linear"),
    "ionization_rate_per_m3_s": ("Ionisation rate", "m^-3 s^-1", "log"),
}
SERIES = (("discharge_a", "I_d (mA)", 1e3), ("exit_ion_beam_a", "I_beam (mA)", 1e3), ("electrons", "N_e (macro)", 1.0),
          ("neutral_density_per_m3", "n_g (1e19 m^-3)", 1e-19), ("thrust_total_n", "T_total (uN)", 1e6))
BODY_RGB = (42, 47, 51)
MASK_RGB = (111, 111, 111)
BODY_INDEX, MASK_INDEX = 254, 255      # palette indices reserved for outside-plasma and under-sampled cells

# viridis anchors (matplotlib's colormap, 9 samples) interpolated linearly -> deterministic 254-entry LUT
_VIRIDIS_ANCHORS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37],
], dtype=np.float64)
_DIVERGING_ANCHORS = np.array([[59, 76, 192], [124, 159, 249], [222, 220, 218], [245, 152, 105], [180, 4, 38]], dtype=np.float64)


def lut(kind: str = "viridis", size: int = 254) -> np.ndarray:
    anchors = _VIRIDIS_ANCHORS if kind == "viridis" else _DIVERGING_ANCHORS
    x = np.linspace(0.0, 1.0, anchors.shape[0])
    t = np.linspace(0.0, 1.0, size)
    return np.stack([np.interp(t, x, anchors[:, c]) for c in range(3)], axis=1).round().astype(np.uint8)


# -- scale ---------------------------------------------------------------------------------

def colour_scale(frames: FrameSet, key: str, plasma_node: np.ndarray, min_samples: int, decades: float = LOG_DECADES) -> dict[str, Any]:
    """Fixed scale over ALL frames on the unmasked plasma nodes (so the eye sees evolution, not autoscaling)."""

    kind = MAP_LABELS[key][2]
    data = frames.maps[key].astype(np.float64)
    valid = plasma_node[None, :, :] & (frames.maps["sample_count_e"] >= min_samples) & np.isfinite(data)
    if key == "ionization_rate_per_m3_s":
        valid &= data > 0.0
    values = data[valid]
    if values.size == 0:
        values = data[plasma_node[None, :, :] & np.isfinite(data)]
    if values.size == 0:
        values = np.array([0.0, 1.0])
    if kind == "log":
        hi = float(np.max(values)) if np.max(values) > 0 else 1.0
        lo = hi / 10.0**decades
        return {"kind": "log", "lo": lo, "hi": hi, "decades": decades}
    if kind == "signed":   # potential: full range over all frames (0 V far field .. anode), diverging palette
        lo, hi = float(np.min(values)), float(np.max(values))
        if not lo < hi:
            lo, hi = lo - 1.0, hi + 1.0
        return {"kind": "signed", "lo": lo, "hi": hi}
    hi = float(np.percentile(values, 99.5)) or 1.0
    return {"kind": "linear", "lo": 0.0, "hi": hi}


# -- ionisation rate: event counts, causal window, resolution mask, robust scale ---------------

def frame_interval_s(frames: FrameSet, dt_s: float) -> np.ndarray:
    """Duration of each recorded interval (s), from the step range of the frame."""

    return (np.asarray(frames.end_step, dtype=np.float64) - np.asarray(frames.start_step, dtype=np.float64)) * float(dt_s)


def ionisation_events(frames: FrameSet, volume_m3: np.ndarray, plasma_node: np.ndarray, macro_weight: float, dt_s: float) -> np.ndarray:
    """Per-frame, per-node macro-ionisation event weight ``rate * V_node * dt_frame / W`` (float64, 0 outside the plasma).

    Same derivation as the dashboards' ``ionization_events`` (commit 6bd5e5b0).  The runner deposits every
    macro-ionisation event with bilinear weights on the four nodes around the born ion, so a node value is a
    fractional share; the sum over the domain is the integer number of events of the interval.
    """

    rate = np.asarray(frames.maps[IZ_KEY], dtype=np.float64)
    volume = np.asarray(volume_m3, dtype=np.float64)
    dt = frame_interval_s(frames, dt_s)
    events = rate * volume[None, :, :] * dt[:, None, None] / float(macro_weight)
    events = np.where(plasma_node[None, :, :], events, 0.0)
    return np.where(np.isfinite(events), events, 0.0)


def causal_window_sum(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing sum over the last ``window`` entries along axis 0 (partial windows at the start; causal)."""

    if window < 1:
        raise ValueError("window must be at least one frame")
    v = np.asarray(values, dtype=np.float64)
    cumulative = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, axis=0)], axis=0)
    n = v.shape[0]
    end = np.arange(1, n + 1)
    start = np.maximum(end - window, 0)
    return cumulative[end] - cumulative[start]


def windowed_ionisation(frames: FrameSet, volume_m3: np.ndarray, plasma_node: np.ndarray, macro_weight: float, dt_s: float,
                        window: int) -> dict[str, Any]:
    """Causal K-frame window of the ionisation map: events, time, rate (= events W / (V T), i.e. the time-weighted mean rate)."""

    events = ionisation_events(frames, volume_m3, plasma_node, macro_weight, dt_s)
    dt = frame_interval_s(frames, dt_s)
    w_events = causal_window_sum(events, window)
    w_time = causal_window_sum(dt, window)
    w_samples = causal_window_sum(np.asarray(frames.maps["sample_count_e"], dtype=np.float64), window)
    n = frames.count
    frames_in_window = np.minimum(np.arange(n) + 1, window)
    volume = np.where(plasma_node, np.asarray(volume_m3, dtype=np.float64), np.inf)
    rate = w_events * float(macro_weight) / (volume[None, :, :] * w_time[:, None, None])
    rate = np.where(plasma_node[None, :, :], rate, 0.0)
    return {"window": int(window), "events": w_events, "rate": rate, "window_s": w_time, "frames_in_window": frames_in_window,
            "sample_count_e": w_samples, "frame_events": events}


def choose_window(events: np.ndarray, sample_count_e: np.ndarray, plasma_node: np.ndarray, *, target_median_events: float = IZ_TARGET_MEDIAN_EVENTS,
                  min_samples: int = MIN_SAMPLES_DEFAULT, max_window: int | None = None) -> int:
    """Smallest K for which the median windowed event count of the resolved, event-bearing nodes reaches the target.

    "Resolved" = at least ``min_samples`` electron samples summed over the window; "event-bearing" = at least
    one event weight in the window.  Full windows only (K..n); the median is taken over all (window, node)
    pairs.  Returns n (the whole run) when no K satisfies the target.
    """

    ev = np.asarray(events, dtype=np.float64)[:, plasma_node]
    es = np.asarray(sample_count_e, dtype=np.float64)[:, plasma_node]
    n = ev.shape[0]
    limit = n if max_window is None else max(1, min(int(max_window), n))
    c_ev = np.concatenate([np.zeros((1, ev.shape[1])), np.cumsum(ev, axis=0)], axis=0)
    c_es = np.concatenate([np.zeros((1, es.shape[1])), np.cumsum(es, axis=0)], axis=0)
    for k in range(1, limit + 1):
        end = np.arange(k, n + 1)
        w_ev = c_ev[end] - c_ev[end - k]
        w_es = c_es[end] - c_es[end - k]
        selected = w_ev[(w_es >= min_samples) & (w_ev >= 1.0)]
        if selected.size and float(np.median(selected)) >= target_median_events:
            return k
    return limit


def ionisation_scale(rate: np.ndarray, events: np.ndarray, plasma_node: np.ndarray, min_events: float,
                     percentiles: tuple[float, float] = IZ_PERCENTILES) -> dict[str, Any]:
    """Fixed log10 scale of the windowed rate between two robust percentiles of the resolved nodes over the whole run."""

    values = np.asarray(rate, dtype=np.float64)
    valid = plasma_node[None, :, :] & (np.asarray(events) >= min_events) & np.isfinite(values) & (values > 0.0)
    picked = values[valid]
    if picked.size == 0:   # nothing resolved: fall back to every positive value so the render still has a scale
        picked = values[plasma_node[None, :, :] & np.isfinite(values) & (values > 0.0)]
    if picked.size == 0:
        picked = np.array([1.0, 10.0])
    lo, hi = (float(v) for v in np.percentile(picked, percentiles))
    if not lo < hi:   # degenerate (one distinct value): one decade either side
        lo, hi = (hi / 10.0, hi * 10.0) if hi > 0 else (0.1, 1.0)
    return {"kind": "log", "lo": lo, "hi": hi, "decades": math.log10(hi / lo), "percentiles": [float(p) for p in percentiles],
            "basis": f"{percentiles[0]:g}th-{percentiles[1]:g}th percentile of the resolved windowed nodes over the run"}


def prepare_ionisation(frames: FrameSet, masks: Any, macro_weight: float, dt_s: float, *, window: int | None = None,
                       min_events: float = MIN_SAMPLES_DEFAULT, min_samples: int = MIN_SAMPLES_DEFAULT,
                       target_median_events: float = IZ_TARGET_MEDIAN_EVENTS, percentiles: tuple[float, float] = IZ_PERCENTILES) -> dict[str, Any]:
    """Everything the two render paths need for the ionisation panel (window, maps, mask, scale, per-frame honesty numbers)."""

    plasma = masks.plasma_node
    volume = np.asarray(masks.shape_volume_m3, dtype=np.float64)
    events = ionisation_events(frames, volume, plasma, macro_weight, dt_s)
    auto = window is None
    k = choose_window(events, frames.maps["sample_count_e"], plasma, target_median_events=target_median_events, min_samples=min_samples) if auto else int(window)
    w = windowed_ionisation(frames, volume, plasma, macro_weight, dt_s, k)
    scale = ionisation_scale(w["rate"], w["events"], plasma, min_events, percentiles)
    resolved = plasma[None, :, :] & (w["events"] >= min_events)
    total = w["events"].sum(axis=(1, 2))
    share = np.where(total > 0, (w["events"] * resolved).sum(axis=(1, 2)) / np.where(total > 0, total, 1.0), np.nan)
    s_window = total * float(macro_weight) / w["window_s"]                       # mean ionisation rate of the window (s^-1)
    bearing = resolved & (w["events"] >= 1.0)
    median_events = float(np.median(w["events"][bearing])) if bearing.any() else float("nan")
    dt = frame_interval_s(frames, dt_s)
    w.update({
        "auto": auto, "min_events": float(min_events), "min_samples": int(min_samples), "target_median_events": float(target_median_events),
        "scale": scale, "resolved": resolved, "resolved_nodes": resolved.sum(axis=(1, 2)), "plasma_nodes": int(plasma.sum()),
        "share_resolved": share, "s_window_per_s": s_window, "median_events_resolved": median_events,
        "nominal_window_s": float(k * np.median(dt)) if frames.count else 0.0,
    })
    return w


def ionisation_legend(iz: Mapping[str, Any], frame_i: int) -> tuple[str, list[str]]:
    """Title suffix and legend lines of the ionisation panel for one frame (ASCII only, see compose_video_frame)."""

    m = int(iz["frames_in_window"][frame_i])
    k = int(iz["window"])
    scale = iz["scale"]
    window_txt = f"window {m}/{k} frames = {iz['window_s'][frame_i] * 1e9:.0f} ns (causal{', partial' if m < k else ''})"
    line1 = (f"log10 {scale['lo']:.2e} - {scale['hi']:.2e} m^-3 s^-1 (fixed over the run: {scale['percentiles'][0]:g}-{scale['percentiles'][1]:g} pct of resolved "
             f"windowed nodes, {scale['decades']:.1f} decades); dark: thruster body")
    line2 = (f"grey = unresolved: < {iz['min_events']:g} macro-ionisation events (bilinear node weight, W = macro weight) in the window; "
             f"no spatial smoothing")
    share = iz["share_resolved"][frame_i]
    share_txt = f"{share * 100:.0f} %" if np.isfinite(share) else "-"
    line3 = (f"resolved: {int(iz['resolved_nodes'][frame_i])} of {iz['plasma_nodes']} plasma nodes carry {share_txt} of the window's ionisation; "
             f"S_window = {iz['s_window_per_s'][frame_i]:.2e} /s")
    return window_txt, [line1, line2, line3]


def index_frame(values: np.ndarray, sample_count: np.ndarray, plasma_node: np.ndarray, scale: Mapping[str, Any], min_samples: int) -> np.ndarray:
    """Map one frame to palette indices 0..253 (colour), 254 (body / outside the plasma), 255 (masked / undefined).

    Radial index increases upward in the image (row 0 = r_max), axial index left to right."""

    v = np.asarray(values, dtype=np.float64)
    if scale["kind"] == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (np.log10(v) - math.log10(scale["lo"])) / (math.log10(scale["hi"]) - math.log10(scale["lo"]))
        undefined = ~np.isfinite(v) | (v <= 0.0)
    else:
        t = (v - scale["lo"]) / (scale["hi"] - scale["lo"] or 1.0)
        undefined = ~np.isfinite(v)
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    idx = np.rint(t * 253.0).astype(np.uint8)
    masked = (np.asarray(sample_count) < min_samples) | undefined
    idx = np.where(masked, MASK_INDEX, idx).astype(np.uint8)
    idx = np.where(plasma_node, idx, BODY_INDEX).astype(np.uint8)
    return idx[::-1, :]


def palette(scale_kind: str) -> np.ndarray:
    table = lut("diverging" if scale_kind == "signed" else "viridis")
    return np.concatenate([table, np.array([BODY_RGB], dtype=np.uint8), np.array([MASK_RGB], dtype=np.uint8)], axis=0)


def to_rgb(idx: np.ndarray, scale_kind: str) -> np.ndarray:
    return palette(scale_kind)[idx]


def downsample(values: np.ndarray, factor: int, *, how: str = "mean") -> np.ndarray:
    """Block reduce a 2-D node map (trailing partial blocks dropped); sample counts are summed."""

    if factor <= 1:
        return np.asarray(values)
    h, w = values.shape
    hh, ww = h // factor * factor, w // factor * factor
    block = np.asarray(values[:hh, :ww], dtype=np.float64).reshape(hh // factor, factor, ww // factor, factor)
    if how == "sum":
        return block.sum(axis=(1, 3))
    if how == "any":
        return block.max(axis=(1, 3)) > 0
    with np.errstate(invalid="ignore"):
        return np.nanmean(block, axis=(1, 3))


# -- geometry / overlay ----------------------------------------------------------------------

def grid_from_summary(summary: Mapping[str, Any]) -> Grid2D:
    cfg = summary["provenance"]["config"]
    g = cfg["grid"]
    geo = g["geometry"]
    geometry = ChannelGeometry(
        geo["bore_radius_m"], geo["z_min_m"], geo["z_max_m"], geo["cone_start_z_m"], geo["exit_radius_m"],
        plume_radius_m=geo.get("plume_radius_m"), plume_length_m=geo.get("plume_length_m"),
        body_dielectric_radius_m=geo.get("body_dielectric_radius_m"),
    )
    return Grid2D(geometry, int(g["radial_cells"]), int(g["axial_cells"]))


def mask_outline(plasma_node: np.ndarray) -> list[list[list[int]]]:
    """Unit-square edges between plasma and non-plasma nodes in (col, row_from_top) pixel coordinates."""

    m = np.asarray(plasma_node, dtype=bool)[::-1, :]
    segments: list[list[list[int]]] = []
    horizontal = m[1:, :] != m[:-1, :]
    for i, j in zip(*np.nonzero(horizontal)):
        segments.append([[int(j), int(i) + 1], [int(j) + 1, int(i) + 1]])
    vertical = m[:, 1:] != m[:, :-1]
    for i, j in zip(*np.nonzero(vertical)):
        segments.append([[int(j) + 1, int(i)], [int(j) + 1, int(i) + 1]])
    return segments


# -- image composition (video) ----------------------------------------------------------------

def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def compose_video_frame(idx: np.ndarray, scale: Mapping[str, Any], key: str, frame_i: int, frames: FrameSet, grid: Grid2D, *,
                        upscale: int, min_samples: int, cusp_z_m: Sequence[float] = (), title_suffix: str | None = None,
                        legend: Sequence[str] | None = None) -> np.ndarray:
    """One RGB video frame: the map (nearest upscale), the body outline, cusp planes, texts and a time-series strip.

    ``title_suffix`` / ``legend`` override the default single-line legend (used by the windowed ionisation panel,
    which declares its window, mask and scale); one extra 16 px row per additional legend line."""

    from PIL import Image, ImageDraw

    rgb = to_rgb(idx, scale["kind"])
    h, w = idx.shape
    map_img = Image.fromarray(rgb, "RGB").resize((w * upscale, h * upscale), Image.NEAREST)
    strip_h = 150
    legend_lines = list(legend) if legend is not None else None
    margin_top = 48 + 16 * (len(legend_lines) - 1 if legend_lines else 0)
    canvas = Image.new("RGB", (map_img.width + 90, margin_top + map_img.height + strip_h + 20), (17, 20, 23))
    canvas.paste(map_img, (0, margin_top))
    draw = ImageDraw.Draw(canvas)
    # body outline (plasma-mask boundary) and cusp planes
    for (x0, y0), (x1, y1) in mask_outline(_plasma(grid)):
        draw.line([(x0 * upscale, margin_top + y0 * upscale), (x1 * upscale, margin_top + y1 * upscale)], fill=(155, 184, 176), width=1)
    z0, z1 = grid.geometry.z_min_m, grid.geometry.domain_z_max_m
    for z in cusp_z_m:
        x = round((z - z0) / (z1 - z0) * (w - 1) * upscale)
        for y in range(margin_top, margin_top + map_img.height, 8):
            draw.line([(x, y), (x, y + 4)], fill=(255, 207, 103), width=1)
    label, unit, kind = MAP_LABELS[key]
    t_us = frames.time_s[frame_i] * 1e6
    font = _font(16)
    small = _font(12)
    # ASCII only: Pillow's fallback bitmap font has no glyphs for the dash / micro sign
    title = f"{label} ({unit}) - t = {t_us:.3f} us, steps {int(frames.start_step[frame_i])}-{int(frames.end_step[frame_i])}"
    if title_suffix:
        title = f"{title}; {title_suffix}"
    draw.text((6, 4), title, fill=(232, 236, 239), font=font)
    if legend_lines is None:
        scale_txt = (f"log10 {scale['lo']:.2e} - {scale['hi']:.2e} (fixed, {scale['decades']:.0f} decades)" if kind == "log"
                     else f"{scale['lo']:.3g} - {scale['hi']:.3g} (fixed)")
        legend_lines = [f"{scale_txt}; grey: < {min_samples} electron samples in the interval; dark: thruster body"]
    for line_i, line in enumerate(legend_lines):
        draw.text((6, 26 + 16 * line_i), line, fill=(155, 184, 176), font=small)
    # colour bar
    bar = palette(kind)[:254][::-1]
    bar_img = Image.fromarray(np.repeat(bar[:, None, :], 16, axis=1), "RGB").resize((16, map_img.height), Image.NEAREST)
    canvas.paste(bar_img, (map_img.width + 20, margin_top))
    hi_txt = f"{scale['hi']:.1e}" if kind == "log" else f"{scale['hi']:.3g}"
    lo_txt = f"{scale['lo']:.1e}" if kind == "log" else f"{scale['lo']:.3g}"
    draw.text((map_img.width + 40, margin_top), hi_txt, fill=(232, 236, 239), font=small)
    draw.text((map_img.width + 40, margin_top + map_img.height - 14), lo_txt, fill=(232, 236, 239), font=small)
    # time-series strip with the cursor
    top = margin_top + map_img.height + 10
    width = map_img.width
    draw.rectangle([(0, top), (width, top + strip_h)], fill=(24, 28, 32))
    n = frames.count
    xs = [round(k / max(n - 1, 1) * (width - 1)) for k in range(n)]
    colours = [(255, 107, 107), (90, 214, 192), (88, 168, 255), (255, 207, 103), (200, 160, 255)]
    texts = []
    for (skey, slabel, factor), colour in zip(SERIES, colours):
        y = frames.scalars[skey] * factor
        finite = np.isfinite(y)
        if not finite.any():
            continue
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        span = hi - lo or 1.0
        pts = [(xs[k], top + strip_h - 8 - int((y[k] - lo) / span * (strip_h - 30))) for k in range(n) if finite[k]]
        if len(pts) > 1:
            draw.line(pts, fill=colour, width=1)
        value = y[frame_i]
        texts.append((f"{slabel}: {value:.4g}" if np.isfinite(value) else f"{slabel}: -", colour))
    cx = xs[frame_i]
    draw.line([(cx, top), (cx, top + strip_h)], fill=(232, 236, 239), width=1)
    x_text = 6
    for text, colour in texts:
        draw.text((x_text, top + 2), text, fill=colour, font=small)
        x_text += int(draw.textlength(text, font=small)) + 14
    return np.asarray(canvas)


_MASKS_CACHE: dict[int, Any] = {}


def _masks(grid: Grid2D) -> Any:
    key = id(grid)
    if key not in _MASKS_CACHE:
        _MASKS_CACHE[key] = build_mesh_masks(grid)
    return _MASKS_CACHE[key]


def _plasma(grid: Grid2D) -> np.ndarray:
    return _masks(grid).plasma_node


def run_constants(summary: Mapping[str, Any]) -> tuple[float, float]:
    """(macro weight W, time step dt_s) of the run, from the recorded configuration."""

    cfg = summary["provenance"]["config"]
    return float(cfg["macro_weight"]), float(cfg["dt_s"])


# -- video writers ---------------------------------------------------------------------------

def available_backend() -> str:
    try:
        import imageio_ffmpeg  # noqa: F401
        return "imageio_ffmpeg"
    except ImportError:
        pass
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    return "pillow_gif"


def write_video(images: Sequence[np.ndarray], path: Path, fps: int, backend: str | None = None) -> tuple[Path, str]:
    """Write the RGB frames; returns (path actually written, backend). MP4 needs even dimensions (padded)."""

    backend = backend or available_backend()
    frames = [np.asarray(im, dtype=np.uint8) for im in images]
    h, w = frames[0].shape[:2]
    if backend in ("imageio_ffmpeg", "ffmpeg"):
        if h % 2 or w % 2:
            frames = [np.pad(f, ((0, h % 2), (0, w % 2), (0, 0)), constant_values=17) for f in frames]
            h, w = frames[0].shape[:2]
        path = path.with_suffix(".mp4")
        if backend == "imageio_ffmpeg":
            import imageio_ffmpeg
            writer = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                                 output_params=["-crf", "18", "-preset", "medium"], macro_block_size=1)
            writer.send(None)
            for f in frames:
                writer.send(np.ascontiguousarray(f).tobytes())
            writer.close()
        else:
            cmd = [shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                   "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                   "-movflags", "+faststart", str(path)]
            proc = subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames), capture_output=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[-2000:]}")
        return path, backend
    from PIL import Image
    path = path.with_suffix(".gif")
    pil = [Image.fromarray(f, "RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=round(1000 / fps), loop=0, optimize=False)
    return path, "pillow_gif"


# -- HTML player -----------------------------------------------------------------------------

def _png_bytes(idx: np.ndarray, scale_kind: str) -> bytes:
    from PIL import Image
    image = Image.fromarray(idx, "P")
    image.putpalette(palette(scale_kind).reshape(-1).tolist())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _round(values: np.ndarray, digits: int = 6) -> list[float | None]:
    return [None if not np.isfinite(v) else float(f"{v:.{digits}g}") for v in np.asarray(values, dtype=np.float64)]


def ionisation_payload_block(iz: Mapping[str, Any]) -> dict[str, Any]:
    """Window / mask / scale declaration of the ionisation map for the player (what the viewer is looking at)."""

    scale = iz["scale"]
    return {
        "window": {"frames": int(iz["window"]), "seconds": float(f"{iz['nominal_window_s']:.6g}"), "causal": True, "auto": bool(iz["auto"]),
                   "target_median_events": float(iz["target_median_events"]), "median_events_resolved": _round(np.array([iz["median_events_resolved"]]))[0],
                   "frames_in_window": [int(v) for v in iz["frames_in_window"]], "window_s": _round(iz["window_s"], 6),
                   "resolved_nodes": [int(v) for v in iz["resolved_nodes"]], "plasma_nodes": int(iz["plasma_nodes"]),
                   "share_resolved": _round(iz["share_resolved"], 4), "s_window_per_s": _round(iz["s_window_per_s"], 5)},
        "mask": {"counts": "macro-ionisation events (bilinear node weight, rate V_node T_window / W)", "min_count": float(iz["min_events"]),
                 "note": (f"grey = unresolved: fewer than {iz['min_events']:g} macro-ionisation events in the causal {iz['window']}-frame window "
                          f"({iz['nominal_window_s'] * 1e9:.0f} ns; partial over the first {iz['window'] - 1} frames); no spatial smoothing")},
        "scale_note": (f"log10 {scale['lo']:.2e} - {scale['hi']:.2e} m^-3 s^-1, fixed over the run at the {scale['percentiles'][0]:g}th-"
                       f"{scale['percentiles'][1]:g}th percentile of the resolved windowed nodes ({scale['decades']:.1f} decades), not the per-frame maximum"),
    }


def build_player_payload(results: Path, frames: FrameSet, summary: Mapping[str, Any], grid: Grid2D, *, maps: Sequence[str] = DEFAULT_MAPS,
                         min_samples: int = MIN_SAMPLES_DEFAULT, factor: int = 2, cusp_z_m: Sequence[float] = (),
                         ionisation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    plasma = _plasma(grid)
    plasma_ds = downsample(plasma.astype(np.float64), factor, how="any") if factor > 1 else plasma
    if IZ_KEY in maps and ionisation is None:
        macro_weight, dt_s = run_constants(summary)
        ionisation = prepare_ionisation(frames, _masks(grid), macro_weight, dt_s, min_samples=min_samples)
    scales: dict[str, Any] = {}
    images: dict[str, list[str]] = {}
    extras: dict[str, dict[str, Any]] = {}
    for key in maps:
        if key == IZ_KEY:
            assert ionisation is not None
            scale = ionisation["scale"]
            value_maps, count_maps, threshold = ionisation["rate"], ionisation["events"], float(ionisation["min_events"])
            extras[key] = ionisation_payload_block(ionisation)
        else:
            scale = colour_scale(frames, key, plasma, min_samples)
            value_maps, count_maps, threshold = frames.maps[key], frames.maps["sample_count_e"], float(min_samples)
            extras[key] = {"mask": {"counts": "electron samples", "min_count": float(min_samples),
                                    "note": f"grey: fewer than {min_samples} electron samples in the frame interval"}}
        scales[key] = scale
        pngs = []
        for i in range(frames.count):
            values = downsample(value_maps[i], factor) if factor > 1 else value_maps[i]
            counts = downsample(count_maps[i], factor, how="sum") if factor > 1 else count_maps[i]
            idx = index_frame(values, counts, plasma_ds, scale, threshold)
            pngs.append(base64.b64encode(_png_bytes(idx, scale["kind"])).decode("ascii"))
        images[key] = pngs
    geometry = grid.geometry
    manifest = (summary.get("artifacts") or {}).get("frames") or {}
    return {
        "schema": SCHEMA,
        "experiment_id": summary.get("experiment_id"),
        "model_version": summary.get("model_version"),
        "status": summary.get("status"),
        "claim_boundary": summary.get("claim_boundary"),
        "claim_statement": (
            _claim_lead(summary.get("status")) +
            "not a thruster performance prediction. Each frame averages the recorded interval; the colour scale is fixed across all "
            "frames (log with a floor for the densities; log between two robust percentiles for the ionisation rate) so evolution is "
            "visible without autoscaling; grey cells were sampled by fewer macro-electrons than the declared threshold in that interval. "
            "The ionisation-rate map is a causal rolling window of recorded frames (declared in its caption) and greys out nodes with "
            "fewer macro-ionisation events than its threshold: a single frame is shot-noise dominated at this cadence."
        ),
        "protocol_sha256": summary.get("protocol_sha256"),
        "frames_sha256": manifest.get("sha256"),
        "frame_count": frames.count,
        "cadence_steps": int(frames.end_step[0] - frames.start_step[0]) if frames.count else None,
        "interval_s": float(frames.time_s[0] / max(int(frames.end_step[0]), 1) * (frames.end_step[0] - frames.start_step[0])) if frames.count else None,
        "downsample_factor": factor,
        "min_samples": min_samples,
        "min_samples_note": (f"grey: fewer than {min_samples} electron samples in the frame interval (summed over the {factor}x{factor} block)"
                             if factor > 1 else f"grey: fewer than {min_samples} electron samples in the frame interval"),
        "precision": frames.precision,
        "time_s": _round(frames.time_s, 8),
        "start_step": [int(v) for v in frames.start_step],
        "end_step": [int(v) for v in frames.end_step],
        "series": {key: {"label": label, "values": _round(frames.scalars[key] * factor_)} for key, label, factor_ in SERIES},
        "maps": {key: {"label": MAP_LABELS[key][0], "unit": MAP_LABELS[key][1], "scale": scales[key], "png": images[key], **extras[key]} for key in maps},
        "image_shape": [int(plasma_ds.shape[0]), int(plasma_ds.shape[1])],
        "domain": {"z_min_m": geometry.z_min_m, "z_max_m": geometry.domain_z_max_m, "r_max_m": geometry.max_radius_m,
                   "channel_z_max_m": geometry.z_max_m, "has_plume": bool(geometry.has_plume)},
        "body_outline": mask_outline(plasma_ds),
        "cusp_z_m": [float(z) for z in cusp_z_m],
        "palette": {"viridis": palette("log")[:254].tolist(), "diverging": palette("signed")[:254].tolist(), "body": list(BODY_RGB), "mask": list(MASK_RGB)},
    }


DEVELOPMENT_STATUS = "development_screening_not_preregistered"
PREREGISTERED_STATUS_PREFIX = "preregistered_"


def _is_preregistered(status: Any) -> bool:
    return isinstance(status, str) and status.startswith(PREREGISTERED_STATUS_PREFIX)


def _claim_lead(status: Any) -> str:
    """The first clause of the player's claim statement: a development run is 'not preregistered'; a preregistered run's frames are
    diagnostics of its one execution (the verdict lives in the run's ``assessment.json``, never in a video)."""

    if _is_preregistered(status):
        return ("Time series of interval-averaged maps from the one preregistered execution of the run (frames are diagnostics, not gates; "
                "the predeclared verdict is in the run's assessment record); not validated against experiment, ")
    return "Time series of interval-averaged maps from one development run; not preregistered, not validated against experiment, "


def validate_player_payload(payload: Mapping[str, Any]) -> None:
    if payload["schema"] != SCHEMA:
        raise ValueError("unsupported player schema")
    status = payload["status"]
    if status != DEVELOPMENT_STATUS and not _is_preregistered(status):
        raise ValueError("player payload must carry the development/screening status or a preregistered_* run status")
    required = ("not validated", "not a thruster performance prediction", "fixed across all",
                "preregistered execution" if _is_preregistered(status) else "not preregistered")
    for phrase in required:
        if phrase not in payload["claim_statement"].lower():
            raise ValueError(f"claim statement must say '{phrase}'")
    n = payload["frame_count"]
    if n < 1 or len(payload["time_s"]) != n or len(payload["end_step"]) != n:
        raise ValueError("frame count / time axis mismatch")
    if any(b <= a for a, b in zip(payload["end_step"], payload["end_step"][1:])):
        raise ValueError("frame steps must increase")
    for key, block in payload["maps"].items():
        if len(block["png"]) != n:
            raise ValueError(f"map {key} has {len(block['png'])} frames, expected {n}")
        if key not in MAP_LABELS or block["scale"]["kind"] not in ("log", "signed", "linear"):
            raise ValueError("unknown map or scale")
        if not block["scale"]["lo"] < block["scale"]["hi"]:
            raise ValueError("scale must be increasing")
        mask = block.get("mask")
        if not isinstance(mask, Mapping) or not mask.get("min_count", -1.0) >= 0.0 or "note" not in mask:
            raise ValueError(f"map {key} must declare its resolution mask (counts, min_count, note)")
        if key == IZ_KEY:
            window = block.get("window")
            if not isinstance(window, Mapping) or int(window.get("frames", 0)) < 1 or window.get("causal") is not True:
                raise ValueError("the ionisation map must declare a causal window of at least one frame")
            if len(window["frames_in_window"]) != n or any(not 1 <= m <= window["frames"] for m in window["frames_in_window"]):
                raise ValueError("ionisation window lengths must be 1..K for every frame")
            if any(m != min(i + 1, window["frames"]) for i, m in enumerate(window["frames_in_window"])):
                raise ValueError("ionisation window must be causal: min(i + 1, K) frames at frame i")
            if "percentiles" not in block["scale"] or "scale_note" not in block:
                raise ValueError("the ionisation scale must declare its percentile basis")
    for key, block in payload["series"].items():
        if len(block["values"]) != n:
            raise ValueError(f"series {key} length mismatch")
    if not isinstance(payload["frames_sha256"], (str, type(None))):
        raise TypeError("frames_sha256 must be a hash or null")
    if payload["min_samples"] < 0 or payload["downsample_factor"] < 1:
        raise ValueError("bad mask / downsample declaration")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{--bg:#0f1418;--panel:#161c21;--text:#e8ecef;--muted:#9bb8b0;--accent:#5ad6c0;--border:#243038}
body{margin:0;font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text);padding:1rem}
h1{font-size:1.05rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:.2rem 0 .6rem}
.claim{border:1px solid #6a5a24;background:#1f1a0f;border-radius:8px;padding:.6rem .9rem;margin-bottom:.8rem;color:#f0d98a}
.controls{display:flex;flex-wrap:wrap;gap:.8rem;align-items:center;margin:.4rem 0}
.controls label{color:var(--muted);font-size:.85rem;display:block}
select,button,input[type=range]{background:#1d252b;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:.3rem .5rem}
button{cursor:pointer}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:.8rem}
canvas{display:block;width:100%;background:#111}
.small{color:var(--muted);font-size:.82rem}
.grid{display:grid;grid-template-columns:1fr;gap:.8rem}
.kv{display:grid;grid-template-columns:auto 1fr;gap:.2rem .8rem;font-size:.85rem}
.kv span:nth-child(odd){color:var(--muted)}
</style></head><body>
<h1>PIC-MCC · axisymmetric (r,z) · time series · __TITLE__</h1>
<div class="claim" id="claim"></div>
<div class="controls">
<div><label for="map">Map</label><select id="map"></select></div>
<div><label>&nbsp;</label><button id="play" aria-label="play or pause">▶ play</button></div>
<div><label for="speed">Speed (frames/s)</label><select id="speed"><option>4</option><option selected>10</option><option>20</option><option>40</option></select></div>
<div style="flex:1;min-width:260px"><label for="frame">Frame <span id="frameLabel"></span></label><input id="frame" type="range" min="0" value="0" step="1" style="width:100%"></div>
</div>
<div class="grid">
<div class="panel"><canvas id="view" tabindex="0" role="img" aria-label="Interval-averaged map over the (r,z) domain with the thruster body and cusp planes drawn"></canvas><p class="small" id="caption"></p></div>
<div class="panel"><canvas id="series" role="img" aria-label="Synchronised time series with a cursor at the current frame"></canvas><p class="small" id="seriesCaption"></p></div>
<div class="panel"><h2 style="font-size:1rem;margin:.2rem 0 .5rem">Run</h2><div class="kv" id="details"></div></div>
</div>
<script id="pic2d-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("pic2d-data").textContent);
const $=id=>document.getElementById(id);
const mapKeys=Object.keys(DATA.maps);let mapKey=mapKeys[0],frame=0,playing=false,timer=null;
const images={};for(const k of mapKeys){images[k]=DATA.maps[k].png.map(b=>{const im=new Image();im.src="data:image/png;base64,"+b;return im})}
$("claim").textContent=DATA.claim_statement+(DATA.claim_boundary?" "+DATA.claim_boundary:"");
for(const k of mapKeys){const o=document.createElement("option");o.value=k;o.textContent=`${DATA.maps[k].label} (${DATA.maps[k].unit})`;$("map").appendChild(o)}
$("frame").max=String(DATA.frame_count-1);
const fmt=(v,d=3)=>v==null||!isFinite(v)?"–":Number(v).toPrecision(d);
const sci=(v,d=2)=>v==null||!isFinite(v)?"–":Number(v).toExponential(d);
function setup(c,ratio){const w=c.clientWidth||900,h=Math.round(w*ratio),dpr=window.devicePixelRatio||1;c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);const g=c.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);return {g,w,h}}
function drawView(){const c=$("view"),[H,W]=DATA.image_shape,{g,w,h}=setup(c,(H/W)*0.9+0.06);const l=58,t=8,r=92,b=34,pw=w-l-r,ph=h-t-b;g.clearRect(0,0,w,h);g.fillStyle="#111";g.fillRect(0,0,w,h);
const im=images[mapKey][frame];g.imageSmoothingEnabled=false;if(im.complete&&im.naturalWidth)g.drawImage(im,l,t,pw,ph);else im.onload=drawView;
const sx=pw/W,sy=ph/H;g.strokeStyle="#9bb8b0";g.lineWidth=1;g.beginPath();for(const [[x0,y0],[x1,y1]] of DATA.body_outline){g.moveTo(l+x0*sx,t+y0*sy);g.lineTo(l+x1*sx,t+y1*sy)}g.stroke();
const D=DATA.domain;g.setLineDash([4,4]);g.strokeStyle="#ffcf67";for(const z of DATA.cusp_z_m){const x=l+(z-D.z_min_m)/(D.z_max_m-D.z_min_m)*pw;g.beginPath();g.moveTo(x,t);g.lineTo(x,t+ph);g.stroke()}g.setLineDash([]);
g.fillStyle="#e8ecef";g.font="12px system-ui";g.textAlign="center";for(let k=0;k<=4;k++){const z=D.z_min_m+k/4*(D.z_max_m-D.z_min_m);g.fillText((z*1e3).toFixed(1),l+k/4*pw,h-14)}g.fillText("z (mm)",l+pw/2,h-2);
g.textAlign="right";for(let k=0;k<=3;k++){const rr=k/3*D.r_max_m;g.fillText((rr*1e3).toFixed(1),l-6,t+ph-k/3*ph+4)}g.save();g.translate(12,t+ph/2);g.rotate(-Math.PI/2);g.textAlign="center";g.fillText("r (mm)",0,0);g.restore();
const S=DATA.maps[mapKey].scale,pal=S.kind==="signed"?DATA.palette.diverging:DATA.palette.viridis,bx=l+pw+14;for(let k=0;k<ph;k++){const c2=pal[Math.min(253,Math.floor((1-k/ph)*253))];g.fillStyle=`rgb(${c2[0]},${c2[1]},${c2[2]})`;g.fillRect(bx,t+k,12,1.5)}
const M=DATA.maps[mapKey],mk=M.mask||{min_count:DATA.min_samples},Wn=M.window;
g.textAlign="left";g.fillStyle="#e8ecef";g.fillText(S.kind==="log"?sci(S.hi,1):fmt(S.hi,3),bx+16,t+10);g.fillText(S.kind==="log"?sci(S.lo,1):fmt(S.lo,3),bx+16,t+ph);g.fillStyle=`rgb(${DATA.palette.mask.join(",")})`;g.fillRect(bx,t+ph+8,12,8);g.fillStyle="#9bb8b0";g.font="11px system-ui";g.fillText(`< ${mk.min_count}`,bx+16,t+ph+15);
const tUs=DATA.time_s[frame]*1e6,win=Wn?` · window ${Wn.frames_in_window[frame]}/${Wn.frames} frames = ${(Wn.window_s[frame]*1e9).toFixed(0)} ns`:"";$("frameLabel").textContent=`${frame+1} / ${DATA.frame_count} — t = ${tUs.toFixed(3)} µs (steps ${DATA.start_step[frame]}–${DATA.end_step[frame]})${win}`;
const scaleTxt=S.kind==="log"?(S.percentiles?M.scale_note:`log10 ${sci(S.lo,2)} – ${sci(S.hi,2)} (${S.decades} decades, floor = max / 10^${S.decades})`):S.kind==="signed"?`${fmt(S.lo,3)} – ${fmt(S.hi,3)} (full range over all frames)`:`linear 0 – ${fmt(S.hi,3)} (99.5th percentile)`;
const winTxt=Wn?`causal rolling window of ${Wn.frames} frames (${(Wn.seconds*1e9).toFixed(0)} ns${Wn.auto?", chosen so the median resolved event-bearing node holds ≥ "+Wn.target_median_events+" events":""}); this frame: ${Wn.frames_in_window[frame]} frames, ${Wn.resolved_nodes[frame]} of ${Wn.plasma_nodes} plasma nodes resolved carrying ${Wn.share_resolved[frame]==null?"–":(Wn.share_resolved[frame]*100).toFixed(0)+" %"} of the window's ionisation (S_window ${sci(Wn.s_window_per_s[frame],2)} s⁻¹)`:`interval average over ${DATA.cadence_steps} steps = ${(DATA.interval_s*1e9).toFixed(1)} ns`;
$("caption").textContent=`${M.label} (${M.unit}), ${winTxt}; colour scale fixed over all frames: ${scaleTxt}; ${mk.note||DATA.min_samples_note}${DATA.downsample_factor>1?" (counts summed over the "+DATA.downsample_factor+"×"+DATA.downsample_factor+" block)":""}; dark = thruster body / outside the plasma cell mask; dashed verticals = cusp planes; spatial downsample ${DATA.downsample_factor}× (block mean).`}
function drawSeries(){const c=$("series"),{g,w,h}=setup(c,0.28),l=58,t=10,r=16,b=30,pw=w-l-r,ph=h-t-b,keys=Object.keys(DATA.series),cols=["#ff6b6b","#5ad6c0","#58a8ff","#ffcf67","#c8a0ff"],n=DATA.frame_count;g.clearRect(0,0,w,h);g.fillStyle="#161c21";g.fillRect(0,0,w,h);
const x=i=>l+(n>1?i/(n-1):0.5)*pw;let legendX=l;keys.forEach((k,ki)=>{const v=DATA.series[k].values,fin=v.filter(u=>u!=null&&isFinite(u));if(!fin.length)return;const lo=Math.min(...fin),hi=Math.max(...fin),span=hi-lo||1;g.strokeStyle=cols[ki];g.lineWidth=1.4;g.beginPath();let on=false;v.forEach((u,i)=>{if(u==null||!isFinite(u)){on=false;return}const yy=t+ph-(u-lo)/span*ph;on?g.lineTo(x(i),yy):g.moveTo(x(i),yy);on=true});g.stroke();const cur=v[frame];const txt=`${DATA.series[k].label}: ${cur==null?"–":fmt(cur,4)}`;g.fillStyle=cols[ki];g.font="12px system-ui";g.textAlign="left";g.fillText(txt,legendX,h-8);legendX+=g.measureText(txt).width+18});
g.strokeStyle="#e8ecef";g.beginPath();g.moveTo(x(frame),t);g.lineTo(x(frame),t+ph);g.stroke();g.fillStyle="#9bb8b0";g.font="11px system-ui";g.textAlign="center";for(let k=0;k<=4;k++){const i=Math.round(k/4*(n-1));g.fillText((DATA.time_s[i]*1e6).toFixed(2)+" µs",x(i),t+ph+12)}
$("seriesCaption").textContent="Each series is normalised to its own range (values at the cursor in the legend); the cursor is the current frame's end time. I_d = anode electron minus anode ion current; I_beam = far-field ion current; N_e = macro-electrons; n_g = channel neutral density; T_total = momentum-flux thrust + cold-gas thrust (development numbers)."}
function draw(){drawView();drawSeries()}
function setFrame(i){frame=Math.max(0,Math.min(DATA.frame_count-1,i));$("frame").value=String(frame);draw()}
$("frame").addEventListener("input",e=>setFrame(Number(e.target.value)));$("map").addEventListener("change",e=>{mapKey=e.target.value;draw()});
function tick(){if(!playing)return;setFrame(frame+1>=DATA.frame_count?0:frame+1);timer=setTimeout(tick,1000/Number($("speed").value))}
$("play").addEventListener("click",()=>{playing=!playing;$("play").textContent=playing?"❚❚ pause":"▶ play";if(playing)tick();else clearTimeout(timer)});
$("view").addEventListener("keydown",e=>{if(e.key==="ArrowRight"){setFrame(frame+1);e.preventDefault()}if(e.key==="ArrowLeft"){setFrame(frame-1);e.preventDefault()}if(e.key===" "){$("play").click();e.preventDefault()}});
$("details").innerHTML=[["experiment",DATA.experiment_id],["model",DATA.model_version],["status",DATA.status],["frames",`${DATA.frame_count} × ${DATA.cadence_steps} steps, ${DATA.precision}`],["protocol sha256",DATA.protocol_sha256],["frames sha256",DATA.frames_sha256||"–"],["domain",`z ${(DATA.domain.z_min_m*1e3).toFixed(1)}–${(DATA.domain.z_max_m*1e3).toFixed(1)} mm, r ≤ ${(DATA.domain.r_max_m*1e3).toFixed(1)} mm${DATA.domain.has_plume?" (channel + plume)":""}`]].map(([k,v])=>`<span>${k}</span><span>${v}</span>`).join("");
new ResizeObserver(()=>draw()).observe(document.body);draw();
</script></body></html>
"""


def render_player_html(payload: Mapping[str, Any], title: str) -> str:
    validate_player_payload(payload)
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


# -- driver ----------------------------------------------------------------------------------

def load_summary(results: Path) -> dict[str, Any]:
    path = results / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (the run has not written its final artifacts yet)")
    return json.loads(path.read_text(encoding="utf-8"))


def cusp_planes(results: Path, protocol_path: Path | None) -> list[float]:
    """Cusp planes from the steady-state dashboard's field authority (optional; [] if unavailable)."""

    if protocol_path is None:
        return []
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pic2d_steady_dashboard", MODERN / "visualization" / "generate_pic2d_cft_steady_state.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        case = module.build_case(results, protocol_path, label="frames", role="headline")
        return [float(z) for z in case["cusps"]["cusp_z_m"]]
    except Exception:  # noqa: BLE001 - the planes are decoration; the render must not depend on them
        return []


def render_run(results: Path, out_dir: Path, *, maps: Sequence[str] = DEFAULT_MAPS, fps: int = 10, min_samples: int = MIN_SAMPLES_DEFAULT,
               upscale: int = 3, factor: int = 2, backend: str | None = None, cusp_z_m: Sequence[float] | None = None,
               protocol_path: Path | None = None, video: bool = True, html: bool = True, suffix: str = "",
               iz_window: int | None = None, iz_min_events: float = MIN_SAMPLES_DEFAULT, iz_percentiles: tuple[float, float] = IZ_PERCENTILES,
               iz_target_median_events: float = IZ_TARGET_MEDIAN_EVENTS) -> dict[str, Any]:
    """Render the videos and the HTML player.  ``suffix`` is appended to every output name (e.g. ``-v2``).

    Ionisation panel: ``iz_window`` frames (None = automatic, see ``choose_window``), ``iz_min_events`` resolution
    threshold on the windowed event weight, ``iz_percentiles`` fixed colour range of the windowed rate."""

    results = Path(results)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(results)
    frames = load_frames(results)
    grid = grid_from_summary(summary)
    plasma = _plasma(grid)
    cusps = list(cusp_z_m) if cusp_z_m is not None else cusp_planes(results, protocol_path)
    run_name = results.parent.name if results.name == "results" else results.name
    report: dict[str, Any] = {"run": run_name, "frames": frames.count, "videos": {}, "html": None, "backend": None,
                              "scales": {}, "min_samples": min_samples, "upscale": upscale, "downsample_factor": factor, "suffix": suffix}
    ionisation: dict[str, Any] | None = None
    if IZ_KEY in maps:
        macro_weight, dt_s = run_constants(summary)
        ionisation = prepare_ionisation(frames, _masks(grid), macro_weight, dt_s, window=iz_window, min_events=iz_min_events, min_samples=min_samples,
                                        target_median_events=iz_target_median_events, percentiles=iz_percentiles)
        share = ionisation["share_resolved"]
        report["ionisation"] = {
            "window_frames": ionisation["window"], "window_s": ionisation["nominal_window_s"], "auto": ionisation["auto"],
            "target_median_events": ionisation["target_median_events"], "median_events_resolved": ionisation["median_events_resolved"],
            "min_events": ionisation["min_events"], "scale": ionisation["scale"], "macro_weight": macro_weight, "dt_s": dt_s,
            "resolved_fraction_of_plasma_nodes_mean": float(np.mean(ionisation["resolved_nodes"]) / max(ionisation["plasma_nodes"], 1)),
            "share_of_ionisation_resolved_mean": float(np.nanmean(share)) if np.isfinite(share).any() else None,
            "share_of_ionisation_resolved_last": float(share[-1]) if np.isfinite(share[-1]) else None,
        }
    if video:
        for key in maps:
            if key == IZ_KEY:
                assert ionisation is not None
                scale = ionisation["scale"]
                images = []
                for i in range(frames.count):
                    title_suffix, legend = ionisation_legend(ionisation, i)
                    idx = index_frame(ionisation["rate"][i], ionisation["events"][i], plasma, scale, ionisation["min_events"])
                    images.append(compose_video_frame(idx, scale, key, i, frames, grid, upscale=upscale, min_samples=min_samples, cusp_z_m=cusps,
                                                      title_suffix=title_suffix, legend=legend))
            else:
                scale = colour_scale(frames, key, plasma, min_samples)
                images = [compose_video_frame(index_frame(frames.maps[key][i], frames.maps["sample_count_e"][i], plasma, scale, min_samples),
                                              scale, key, i, frames, grid, upscale=upscale, min_samples=min_samples, cusp_z_m=cusps)
                          for i in range(frames.count)]
            report["scales"][key] = scale
            path, used = write_video(images, out_dir / f"pic2d-{run_name}-{key}{suffix}", fps, backend)
            report["videos"][key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            report["backend"] = used
    if html:
        payload = build_player_payload(results, frames, summary, grid, maps=maps, min_samples=min_samples, factor=factor, cusp_z_m=cusps,
                                       ionisation=ionisation)
        text = render_player_html(payload, f"{run_name} ({summary.get('model_version', '')})")
        path = out_dir / f"pic2d-{run_name}-timeseries{suffix}.html"
        path.write_text(text, encoding="utf-8", newline="\n")
        report["html"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="results directory with frames/ and summary.json")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default: <results>/video)")
    parser.add_argument("--maps", nargs="*", default=list(DEFAULT_MAPS), choices=list(MAP_LABELS))
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES_DEFAULT)
    parser.add_argument("--upscale", type=int, default=3, help="nearest-neighbour upscale of the video map")
    parser.add_argument("--downsample", type=int, default=2, help="spatial block factor for the HTML player frames")
    parser.add_argument("--backend", choices=["imageio_ffmpeg", "ffmpeg", "pillow_gif"], default=None)
    parser.add_argument("--protocol", type=Path, default=None, help="protocol.json for the cusp planes (optional)")
    parser.add_argument("--cusps", nargs="*", type=float, default=None, help="cusp plane z positions (m); overrides --protocol")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--suffix", default="", help="appended to every output file name, e.g. --suffix=-v2 (keeps an earlier render for comparison)")
    parser.add_argument("--iz-window", type=int, default=None,
                        help=f"ionisation panel: causal window in frames (default: smallest K whose median resolved event-bearing node holds >= {IZ_TARGET_MEDIAN_EVENTS:g} events)")
    parser.add_argument("--iz-min-events", type=float, default=MIN_SAMPLES_DEFAULT,
                        help="ionisation panel: grey out nodes with fewer windowed macro-ionisation events (dashboard mask semantics)")
    parser.add_argument("--iz-percentiles", type=float, nargs=2, default=list(IZ_PERCENTILES), metavar=("LO", "HI"),
                        help="ionisation panel: fixed log colour range = these percentiles of the resolved windowed rate over the run")
    parser.add_argument("--iz-target-median-events", type=float, default=IZ_TARGET_MEDIAN_EVENTS, help="window selection target (auto mode)")
    args = parser.parse_args(argv)
    report = render_run(args.results, args.out or (args.results / "video"), maps=args.maps, fps=args.fps, min_samples=args.min_samples,
                        upscale=args.upscale, factor=args.downsample, backend=args.backend, cusp_z_m=args.cusps, protocol_path=args.protocol,
                        video=not args.no_video, html=not args.no_html, suffix=args.suffix, iz_window=args.iz_window, iz_min_events=args.iz_min_events,
                        iz_percentiles=(float(args.iz_percentiles[0]), float(args.iz_percentiles[1])), iz_target_median_events=args.iz_target_median_events)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
