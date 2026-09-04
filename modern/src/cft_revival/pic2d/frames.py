"""Frame recorder (model v1.4/v2.0 runner): interval-averaged node maps at a declared cadence.

The runner accumulates its window diagnostics as device-side sums that are reset at each
averaging-window boundary.  A frame is the exact average over one cadence interval
[a, b]: the accumulator sums are additive, so ``sums(b) - sums(a)`` is the interval sum and
``DiagnosticAccumulator.from_sums(...).to_arrays`` turns it into the same maps the window
average uses (n_e, n_i, phi, T_e from the moments, ionisation rate, wall/exit/side fluxes,
sample counts, plume histograms).  Nothing is added to the step kernels; a frame costs one
device-to-host copy of the sums.

Alignment contract (validated by the runner): the cadence is a multiple of the sync
interval and divides both ``checkpoint_every_steps`` and ``averaging_window_steps``, so
every checkpoint and every window reset falls on a frame boundary.  Frames are written as
one compressed ``frames/frame-NNNNNN.npz`` each (atomic replace) before the checkpoint that
follows them; on resume, frames past the checkpoint step are removed (``reconcile``), so a
crash between a frame write and its checkpoint cannot duplicate a frame.  The manifest
(count, byte size, SHA-256 over the per-file digests) is hash-bound in ``summary.json``.

Precision: maps are stored as float32 (declared in every frame file); the scalars of the
series record at the frame end are stored as canonical JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .models import PIC2DValidationError
from .simulation import DiagnosticAccumulator, Simulation

FRAME_SCHEMA = "cft-revival.pic2d.frame/1.0.0"
FRAME_DIRNAME = "frames"
FRAME_PATTERN = re.compile(r"^frame-(\d{6})\.npz$")
PRECISIONS = {"float32": np.float32, "float16": np.float16}

# maps stored per frame (the plume histograms and fluxes ride along at negligible size)
MAP_KEYS = ("n_e_per_m3", "n_i_per_m3", "phi_v", "t_e_ev", "ionization_rate_per_m3_s", "sample_count_e")
# v2.4.0 (additive, present only with the Coulomb operator on): window-mean Coulomb frequencies per cell (node layout)
COULOMB_MAP_KEYS = ("coulomb_nu_ee_per_s", "coulomb_nu_ei_per_s", "coulomb_mean_s_ee")
PROFILE_KEYS = (
    "wall_ion_flux_per_m2_s", "wall_electron_flux_per_m2_s", "exit_ion_current_density_a_per_m2",
    "exit_electron_current_density_a_per_m2", "side_ion_current_density_a_per_m2", "side_electron_current_density_a_per_m2",
    "plume_ion_current_per_sr_a", "iedf_ion_counts",
    # v2.2.0 (additive, present only when the wall emits): SEE flux, effective yield and mean emitted energy per wall cell
    "wall_see_flux_per_m2_s", "wall_see_effective_yield", "wall_see_mean_energy_ev",
)
SCALAR_KEYS = (
    "step", "time_s", "electrons", "ions", "discharge_a", "exit_ion_beam_a", "ionization_rate_per_s", "neutral_density_per_m3",
    "thrust_total_n", "thrust_flux_n", "thrust_balance_n", "closure_fraction", "cathode_emission_a", "t_e_mean_ev",
)


@dataclass(frozen=True, slots=True)
class FrameRecorderConfig:
    """Cadence in steps (multiple of the sync interval; divides checkpoint and window), stored precision."""

    cadence_steps: int
    precision: str = "float32"

    def __post_init__(self) -> None:
        if not isinstance(self.cadence_steps, int) or self.cadence_steps <= 0:
            raise PIC2DValidationError("frame cadence must be a positive integer number of steps")
        if self.precision not in PRECISIONS:
            raise PIC2DValidationError(f"frame precision must be one of {sorted(PRECISIONS)}")

    def validate_alignment(self, *, sync_steps: int, checkpoint_every_steps: int, window_steps: int) -> None:
        if self.cadence_steps % sync_steps != 0:
            raise PIC2DValidationError("frame cadence must be a multiple of device_sync_steps")
        if checkpoint_every_steps % self.cadence_steps != 0:
            raise PIC2DValidationError("checkpoint_every_steps must be a multiple of the frame cadence")
        if window_steps % self.cadence_steps != 0:
            raise PIC2DValidationError("averaging_window_steps must be a multiple of the frame cadence")

    def to_dict(self) -> dict[str, Any]:
        return {"cadence_steps": self.cadence_steps, "precision": self.precision, "schema": FRAME_SCHEMA}


def frame_scalars(record: Mapping[str, Any] | None) -> dict[str, float | int | None]:
    """The scalar diagnostics of a series record (flattened) at the frame end."""

    if record is None:
        return {key: None for key in SCALAR_KEYS}
    currents = record.get("currents_a") or {}
    neutral = record.get("neutral") or {}
    momentum = record.get("momentum") or {}
    flat: dict[str, Any] = {
        "step": record.get("step"), "time_s": record.get("time_s"), "electrons": record.get("electrons"), "ions": record.get("ions"),
        "discharge_a": currents.get("discharge_a"), "exit_ion_beam_a": currents.get("exit_ion_beam_a"),
        "ionization_rate_per_s": currents.get("ionization_rate_per_s"), "cathode_emission_a": currents.get("cathode_emission_a"),
        "neutral_density_per_m3": neutral.get("density_per_m3"), "thrust_total_n": momentum.get("thrust_total_n"),
        "thrust_flux_n": momentum.get("thrust_flux_n"), "thrust_balance_n": momentum.get("thrust_balance_n"),
        "closure_fraction": momentum.get("closure_fraction"), "t_e_mean_ev": record.get("t_e_mean_ev"),
    }
    return {key: flat.get(key) for key in SCALAR_KEYS}


def interval_maps(sums_end: Mapping[str, np.ndarray], sums_start: Mapping[str, np.ndarray] | None, masks: Any,
                  macro_weight: float, dt_s: float, iedf_max_ev: float = 450.0) -> dict[str, np.ndarray]:
    """Exact interval average between two cumulative snapshots of the window sums."""

    diff: dict[str, np.ndarray] = {}
    # v2.2.0 / v2.4.0 / v2.5.0: the optional SEE, Coulomb and spatial-neutral sums travel with the others when their option is on
    # (absent otherwise: nothing added to the frame); the neutral sample count rides along like the moment samples
    optional_keys = tuple(key for key in DiagnosticAccumulator.optional_sum_keys() if key in sums_end)
    count_keys = ("neutral_samples",) if "neutral_samples" in sums_end and "neutral_density" in sums_end else ()
    for key in DiagnosticAccumulator.SUM_KEYS + optional_keys + ("steps",) + count_keys:
        end = np.asarray(sums_end[key])
        diff[key] = end.copy() if sums_start is None else end - np.asarray(sums_start[key])
    # v2.0.5: the moment sample count is additive like the sums (absent in pre-v2.0.5 snapshots: one sample per step)
    end_samples = np.asarray(sums_end["moment_samples"] if "moment_samples" in sums_end else sums_end["steps"])
    if sums_start is None:
        diff["moment_samples"] = end_samples.copy()
    else:
        diff["moment_samples"] = end_samples - np.asarray(sums_start["moment_samples"] if "moment_samples" in sums_start else sums_start["steps"])
    steps = int(diff["steps"].reshape(-1)[0])
    if steps <= 0:
        raise PIC2DValidationError("a frame needs a positive number of accumulated steps")
    # (a frame without a moment sample - a cadence shorter than the sampling interval, excluded by the runner's
    #  alignment rule - carries zero sample counts and zero T_e, not an error: the moment maps are ratios)
    return DiagnosticAccumulator.from_sums(masks, diff, iedf_max_ev).to_arrays(macro_weight, dt_s)


def frame_path(results: Path, index: int) -> Path:
    return results / FRAME_DIRNAME / f"frame-{index:06d}.npz"


def list_frames(results: Path) -> list[Path]:
    folder = results / FRAME_DIRNAME
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if FRAME_PATTERN.match(p.name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frames_manifest(results: Path) -> dict[str, Any]:
    """Hash-binding of the frame files: count, bytes, step range, SHA-256 over the sorted per-file digests."""

    files = list_frames(results)
    digest = hashlib.sha256()
    total = 0
    first_end = last_end = None
    for path in files:
        digest.update(path.name.encode("ascii"))
        digest.update(_sha256(path).encode("ascii"))
        total += path.stat().st_size
    if files:
        with np.load(files[0]) as first, np.load(files[-1]) as last:
            first_end = int(first["end_step"][0])
            last_end = int(last["end_step"][0])
    return {
        "count": len(files), "bytes": total, "sha256": digest.hexdigest() if files else None, "directory": FRAME_DIRNAME,
        "first_end_step": first_end, "last_end_step": last_end, "schema": FRAME_SCHEMA,
    }


class FrameRecorder:
    """Writes one frame per cadence interval from the runner loop (see the module docstring)."""

    def __init__(self, results: Path, config: FrameRecorderConfig, sim: Simulation) -> None:
        self.results = Path(results)
        self.config = config
        self.sim = sim
        self.masks = sim.masks
        self.macro_weight = float(sim.config.macro_weight)
        self.dt_s = float(sim.config.dt_s)
        self.iedf_max_ev = float(getattr(sim.backend, "iedf_max_ev", 450.0))
        self._previous: dict[str, np.ndarray] | None = None
        self._previous_step: int = int(sim.backend.step_index)
        (self.results / FRAME_DIRNAME).mkdir(parents=True, exist_ok=True)
        self.index = len(list_frames(self.results))

    # -- lifecycle -------------------------------------------------------------------

    def reconcile(self, checkpoint_step: int) -> int:
        """Drop frames that end after ``checkpoint_step`` (written before a checkpoint that never landed)."""

        removed = 0
        kept: list[Path] = []
        for path in list_frames(self.results):
            with np.load(path) as data:
                end = int(data["end_step"][0])
            if end > checkpoint_step:
                path.unlink()
                removed += 1
            else:
                kept.append(path)
        # frames must be contiguous and end exactly at the checkpoint step boundary of the cadence
        self.index = len(kept)
        self._previous = None
        self._previous_step = int(checkpoint_step)
        return removed

    def on_window_reset(self) -> None:
        """The runner reset the device accumulators (window boundary = frame boundary): restart the differencing."""

        self._previous = None

    def due(self, step: int) -> bool:
        return step % self.config.cadence_steps == 0 and step > self._previous_step

    def steps_to_next_boundary(self, step: int) -> int:
        cadence = self.config.cadence_steps
        return cadence - (step % cadence)

    # -- capture ---------------------------------------------------------------------

    def capture(self, record: Mapping[str, Any] | None) -> Path:
        """Write the frame for (previous boundary, current step]; ``record`` is the series record at the frame end."""

        step = int(self.sim.backend.step_index)
        sums = self.sim.diagnostic_sums()
        maps = interval_maps(sums, self._previous, self.masks, self.macro_weight, self.dt_s, self.iedf_max_ev)
        dtype = PRECISIONS[self.config.precision]
        payload: dict[str, np.ndarray] = {
            "schema": np.array([FRAME_SCHEMA]),
            "precision": np.array([self.config.precision]),
            "start_step": np.array([self._previous_step], dtype=np.int64),
            "end_step": np.array([step], dtype=np.int64),
            "interval_steps": np.array([step - self._previous_step], dtype=np.int64),
            "time_s": np.array([float(self.sim.backend.time_s)]),
            "scalars_json": np.array([json.dumps(frame_scalars(record), sort_keys=True, separators=(",", ":"), allow_nan=False)]),
            "surface_charge_c": np.asarray(self.sim.surface_charge_map(), dtype=dtype),
        }
        for key in MAP_KEYS + PROFILE_KEYS + COULOMB_MAP_KEYS:
            if key in maps:
                payload[key] = np.asarray(maps[key], dtype=dtype)
        if int(maps["window_steps"][0]) != step - self._previous_step:
            raise PIC2DValidationError("frame interval does not match the accumulated steps (window/cadence misalignment)")
        path = frame_path(self.results, self.index)
        tmp = path.with_suffix(".tmp.npz")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(tmp, path)
        self.index += 1
        self._previous = sums
        self._previous_step = step
        return path


# -- reading -----------------------------------------------------------------------------

@dataclass(slots=True)
class FrameSet:
    """All frames of a run loaded into memory (float32 maps stacked along axis 0)."""

    schema: str
    precision: str
    start_step: np.ndarray
    end_step: np.ndarray
    time_s: np.ndarray
    maps: dict[str, np.ndarray]
    profiles: dict[str, np.ndarray]
    surface_charge_c: np.ndarray
    scalars: dict[str, np.ndarray]
    files: list[str]

    @property
    def count(self) -> int:
        return int(self.end_step.size)


def load_frames(results: Path) -> FrameSet:
    files = list_frames(Path(results))
    if not files:
        raise PIC2DValidationError(f"no frames under {results}")
    maps: dict[str, list[np.ndarray]] = {key: [] for key in MAP_KEYS}
    profiles: dict[str, list[np.ndarray]] = {}
    surface: list[np.ndarray] = []
    starts: list[int] = []
    ends: list[int] = []
    times: list[float] = []
    scalars: list[dict[str, Any]] = []
    schema = precision = None
    for path in files:
        with np.load(path) as data:
            schema = str(data["schema"][0]) if schema is None else schema
            precision = str(data["precision"][0]) if precision is None else precision
            if str(data["schema"][0]) != schema or str(data["precision"][0]) != precision:
                raise PIC2DValidationError("frames of one run must share schema and precision")
            starts.append(int(data["start_step"][0]))
            ends.append(int(data["end_step"][0]))
            times.append(float(data["time_s"][0]))
            scalars.append(json.loads(str(data["scalars_json"][0])))
            surface.append(np.asarray(data["surface_charge_c"], dtype=np.float32))
            for key in MAP_KEYS:
                maps[key].append(np.asarray(data[key], dtype=np.float32))
            for key in PROFILE_KEYS + COULOMB_MAP_KEYS:      # optional (SEE / Coulomb) arrays load when present
                if key in data:
                    profiles.setdefault(key, []).append(np.asarray(data[key], dtype=np.float32))
    for a, b in zip(ends, starts[1:]):
        if a != b:
            raise PIC2DValidationError(f"frames are not contiguous: end {a} != next start {b}")
    scalar_arrays = {
        key: np.array([np.nan if s.get(key) is None else float(s[key]) for s in scalars], dtype=np.float64) for key in SCALAR_KEYS
    }
    return FrameSet(
        schema=schema or FRAME_SCHEMA, precision=precision or "float32", start_step=np.array(starts, dtype=np.int64),
        end_step=np.array(ends, dtype=np.int64), time_s=np.array(times, dtype=np.float64),
        maps={key: np.stack(value) for key, value in maps.items()}, profiles={key: np.stack(value) for key, value in profiles.items()},
        surface_charge_c=np.stack(surface), scalars=scalar_arrays, files=[p.name for p in files],
    )


def estimate_frame_bytes(node_shape: tuple[int, int], precision: str = "float32", compression_ratio: float = 1.0) -> int:
    """Uncompressed bytes per frame for the node maps + surface charge (profiles are negligible)."""

    itemsize = np.dtype(PRECISIONS[precision]).itemsize
    nodes = int(node_shape[0]) * int(node_shape[1])
    return int(nodes * itemsize * (len(MAP_KEYS) + 1) / max(compression_ratio, 1e-9))


__all__ = [
    "FRAME_SCHEMA",
    "FrameRecorder",
    "FrameRecorderConfig",
    "FrameSet",
    "COULOMB_MAP_KEYS",
    "MAP_KEYS",
    "PROFILE_KEYS",
    "SCALAR_KEYS",
    "estimate_frame_bytes",
    "frame_scalars",
    "frames_manifest",
    "interval_maps",
    "list_frames",
    "load_frames",
]
