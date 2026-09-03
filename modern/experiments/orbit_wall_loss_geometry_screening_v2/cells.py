"""Catalogue cells, scrambled-Sobol launch design, frozen allocation rule, estimators.

The launch cells of this campaign are the cells of the accepted cusp topology search v3.1
catalogue (``experiments.cusp_topology_search_v3_1.catalogue``): wall intervals between
consecutive separatrix-wall intersections (wall cusps) plus the anode-side and exit-side
partial cells. Every design's cells are bound by the catalogue's sealed bytes, so the
dataset measures wall loss per physical cell.

Per cell the launch design is stratified over the eight ``(energy, pitch, direction)``
strata inherited from v4/v1 (5/25 eV, 20/70 deg, -1/+1). Inside each stratum a scrambled
Sobol' sequence (:mod:`.sobol`) with a frozen per-(design, cell, stratum) seed supplies
the continuous coordinates: radius band selector, radius offset inside the band and
gyrophase. Stage 1 uses points ``0..15`` of every stratum (128 launches per cell); the
frozen allocation rule tops a cell up to points ``0..63`` (512 launches) iff its stage-1
Wilson 95 % width exceeds the threshold. Because the top-up extends the same sequence, the
final 64-point set is a ``(t, 6, 3)``-net per stratum, not an independent sample.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cft_revival.experiment_runtime.canonical import strict_json_file
from cft_revival.orbit_mc import ElectronLaunch, wilson_interval
from cft_revival.orbit_mc.artifacts import content_hash

from experiments.cusp_topology_search_v3_1 import catalogue as catalogue_module

EXPERIMENT = Path(__file__).resolve().parent
MODERN = EXPERIMENT.parents[1]
CATALOGUE_RESULTS = MODERN / "experiments" / "cusp_topology_search_v3_1" / "results"

CELL_POSITION_CLASSES = {
    "anode_partial": "anode_side",
    "interior": "interior",
    "exit_partial": "exit_side",
    "unbounded": "unbounded",
}
WILSON_Z = 1.959963984540054


# --------------------------------------------------------------------------
# Catalogue binding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogueBinding:
    catalogue: Mapping[str, Any]
    file_sha256: str
    manifest_file_sha256: str
    results_root: Path


def load_bound_catalogue(declaration: Mapping[str, Any], results_root: Path = CATALOGUE_RESULTS) -> CatalogueBinding:
    """Load the v3.1 catalogue through its own manifest-bound loader and pin it to the protocol."""

    catalogue = catalogue_module.load_catalogue(results_root)
    raw = (results_root / catalogue_module.CATALOGUE_RELATIVE_PATH).read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    manifest_sha = hashlib.sha256((results_root / "manifest.json").read_bytes()).hexdigest()
    manifest = strict_json_file(results_root / "manifest.json")
    expected = {
        "catalogue_file_sha256": file_sha,
        "manifest_file_sha256": manifest_sha,
        "catalogue_schema_version": catalogue["schema_version"],
        "experiment_id": catalogue["experiment_id"],
        "protocol_semantic_sha256": catalogue["protocol_semantic_sha256"],
        "design_count": catalogue["design_count"],
        "stable_design_count": catalogue["stable_design_count"],
    }
    for key, value in expected.items():
        if declaration[key] != value:
            raise ValueError(f"cusp-cell catalogue authority differs: {key}")
    if manifest.get("state") != "accepted_result" or manifest.get("experiment_id") != declaration["experiment_id"]:
        raise ValueError("the catalogue bundle is not the accepted v3.1 result")
    return CatalogueBinding(catalogue, file_sha, manifest_sha, results_root)


def catalogue_entry(binding: CatalogueBinding, set_id: str, design_id: str) -> Mapping[str, Any]:
    for entry in binding.catalogue["entries"]:
        if entry["set_id"] == set_id and entry["design_id"] == design_id:
            if not entry["stable"]:
                raise ValueError(f"{set_id}:{design_id} is not a stable catalogue entry")
            return entry
    raise KeyError(f"{set_id}:{design_id} is not in the catalogue")


# --------------------------------------------------------------------------
# Cells and launch planes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchCell:
    cell_id: str
    index: int
    kind: str
    position_class: str
    z_start_m: float
    z_end_m: float
    length_m: float
    launch_z_m: float
    launch_plane_inside_injector_zone: bool
    short_cell: bool
    wall_area_m2: float
    start_cusp_id: str | None
    end_cusp_id: str | None
    start_cusp_z_m: float | None
    end_cusp_z_m: float | None
    length_over_pitch: float
    wall_mirror_ratio: float | None
    axis_mirror_ratio: float | None
    wall_b_min_t: float
    cusp_wall_b_min_t: float | None
    axis_bz_peak_t: float
    boundary_ambiguous: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def design_cells(entry: Mapping[str, Any], *, injector_length_m: float, rule: Mapping[str, Any]) -> tuple[LaunchCell, ...]:
    """Catalogue cells of one design with their launch planes under the frozen rule.

    Launch plane: the midpoint of the cell (between its two wall cusps for interior cells;
    between the channel end and the cusp for the partial cells). A midpoint inside the
    injector zone ``[straight_z_min, injector_length]`` is flagged, never moved (one sweep
    design has a 0.16 mm anode-side cell); cells shorter than ``short_cell_length_m`` are
    flagged. Wall areas (the design-pooling weights) use the full catalogue cell length.
    """

    if rule["launch_plane_rule"] != "cell_midpoint":
        raise ValueError("unknown launch plane rule")
    short_length = float(rule["short_cell_length_m"])
    wall_radius = float(entry["geometry"]["wall_radius_m"])
    cusps = {cusp["cusp_id"]: cusp for cusp in entry["wall_cusps"]}
    cells: list[LaunchCell] = []
    for index, cell in enumerate(entry["cells"]):
        kind = cell["kind"]
        if kind not in CELL_POSITION_CLASSES:
            raise ValueError(f"unknown cell kind {kind}")
        z_start = float(cell["z_start_m"])
        z_end = float(cell["z_end_m"])
        if not z_end > z_start:
            raise ValueError(f"{entry['design_id']} {cell['cell_id']}: empty cell")
        launch_z = 0.5 * (z_start + z_end)
        start_cusp = cusps.get(cell["start_cusp_id"]) if cell["start_cusp_id"] else None
        end_cusp = cusps.get(cell["end_cusp_id"]) if cell["end_cusp_id"] else None
        cells.append(
            LaunchCell(
                cell_id=str(cell["cell_id"]),
                index=index,
                kind=kind,
                position_class=CELL_POSITION_CLASSES[kind],
                z_start_m=z_start,
                z_end_m=z_end,
                length_m=float(cell["length_m"]),
                launch_z_m=launch_z,
                launch_plane_inside_injector_zone=bool(launch_z < float(injector_length_m)),
                short_cell=bool((z_end - z_start) < short_length),
                wall_area_m2=2.0 * math.pi * wall_radius * (z_end - z_start),
                start_cusp_id=cell["start_cusp_id"],
                end_cusp_id=cell["end_cusp_id"],
                start_cusp_z_m=None if start_cusp is None else float(start_cusp["z_c_m"]),
                end_cusp_z_m=None if end_cusp is None else float(end_cusp["z_c_m"]),
                length_over_pitch=float(cell["length_over_pitch"]),
                wall_mirror_ratio=cell["wall_mirror_ratio"],
                axis_mirror_ratio=cell["axis_mirror_ratio"],
                wall_b_min_t=float(cell["wall_b_min_t"]),
                cusp_wall_b_min_t=cell["cusp_wall_b_min_t"],
                axis_bz_peak_t=float(cell["axis_bz_peak_t"]),
                boundary_ambiguous=bool(
                    (start_cusp is not None and start_cusp["boundary_ambiguous"])
                    or (end_cusp is not None and end_cusp["boundary_ambiguous"])
                ),
            )
        )
    if "-r" in "".join(item.cell_id for item in cells):
        raise ValueError("cell ids must not contain the '-r' flux-surface separator")
    return tuple(cells)


# --------------------------------------------------------------------------
# Strata and scrambled-Sobol launches
# --------------------------------------------------------------------------


def strata(rule: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    output = []
    for energy_index, energy in enumerate(rule["energies_ev"]):
        for pitch_index, pitch in enumerate(rule["pitch_angles_deg"]):
            for direction in rule["directions"]:
                output.append(
                    {
                        "stratum_id": f"E{energy_index}:P{pitch_index}:D{int(direction):+d}",
                        "energy_index": energy_index,
                        "kinetic_energy_ev": float(energy),
                        "pitch_index": pitch_index,
                        "pitch_angle_deg": float(pitch),
                        "parallel_direction": int(direction),
                    }
                )
    if len(output) != int(rule["strata_per_cell"]):
        raise ValueError("stratum count differs from the declared strata_per_cell")
    return tuple(output)


def stratum_seed(namespace: str, design_key: str, cell_id: str, stratum_id: str) -> int:
    from .sobol import seed_from_bytes

    return seed_from_bytes(f"{namespace}:{design_key}:{cell_id}:{stratum_id}".encode("utf-8"))


def stratum_points(namespace: str, design_key: str, cell_id: str, stratum_id: str, start: int, stop: int) -> np.ndarray:
    from .sobol import scrambled_sobol

    return scrambled_sobol(3, stop - start, stratum_seed(namespace, design_key, cell_id, stratum_id), start=start)


def launch_key(cell_index: int, stratum: Mapping[str, Any], index: int) -> str:
    """Campaign-free physical key in orbit_mc's launch-id grammar ``E<e>:P<p>:X<cell>:D<+-1>:G<index>``.

    orbit_mc v1.7 validates every launch id against ``<campaign_id>:E[0-9]+:P[0-9]+:X[0-9]+:D[+-]1:G[0-9]+``;
    here ``X`` is the catalogue cell index and ``G`` the scrambled-Sobol point index (the
    gyrophase and the radius are functions of that point).
    """

    return f"E{int(stratum['energy_index'])}:P{int(stratum['pitch_index'])}:X{int(cell_index)}:D{int(stratum['parallel_direction']):+d}:G{int(index)}"


def key_of_launch(launch: ElectronLaunch) -> str:
    """Campaign-free physical key of a launch (the last five ':' fields of its id)."""

    parts = launch.launch_id.split(":")
    return ":".join(parts[-5:])


def key_cell_index(key: str) -> int:
    return int(key.split(":")[2][1:])


def key_index(key: str) -> int:
    return int(key.split(":")[4][1:])


def key_stratum_id(key: str) -> str:
    parts = key.split(":")
    return f"{parts[0]}:{parts[1]}:{parts[3]}"


def cell_id_of_key(key: str, cells: Sequence[LaunchCell]) -> str:
    index = key_cell_index(key)
    for cell in cells:
        if cell.index == index:
            return cell.cell_id
    raise KeyError(f"no cell with index {index}")


def _band_and_radius(u_band: float, u_offset: float, bands: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], float]:
    count = len(bands)
    band_index = min(int(math.floor(u_band * count)), count - 1)
    band = bands[band_index]
    centre = float(band["centre_of_wall"])
    half = float(band["half_width_of_wall"])
    return band, centre - half + 2.0 * half * u_offset


def build_launches(
    campaign_id: str,
    *,
    namespace: str,
    design_key: str,
    cells: Sequence[LaunchCell],
    rule: Mapping[str, Any],
    wall_radius_m: float,
    index_ranges: Mapping[str, tuple[int, int]],
    selected_keys: set[str] | None = None,
) -> tuple[ElectronLaunch, ...]:
    """Launches of one case: for every cell in ``index_ranges`` the Sobol indices ``[start, stop)``.

    ``selected_keys`` restricts the result to the given physical keys (control subset).
    Launch id = ``<campaign_id>:E<e>:P<p>:X<cell index>:D<+-1>:G<Sobol index>`` (orbit_mc's
    grammar); seed = first 8 bytes of SHA-256(launch id) (``build_launch_ensemble`` rule).
    """

    bands = rule["radius_bands_of_wall"]
    launches: list[ElectronLaunch] = []
    by_cell = {cell.cell_id: cell for cell in cells}
    for cell_id in sorted(index_ranges):
        cell = by_cell[cell_id]
        start, stop = index_ranges[cell_id]
        if not 0 <= start < stop:
            raise ValueError(f"{cell_id}: invalid index range")
        for stratum in strata(rule):
            points = stratum_points(namespace, design_key, cell_id, stratum["stratum_id"], start, stop)
            for offset, row in enumerate(points):
                index = start + offset
                key = launch_key(cell.index, stratum, index)
                if selected_keys is not None and key not in selected_keys:
                    continue
                band, fraction = _band_and_radius(float(row[0]), float(row[1]), bands)
                identity = f"{campaign_id}:{key}"
                seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
                launches.append(
                    ElectronLaunch(
                        launch_id=identity,
                        seed_id=seed,
                        kinetic_energy_ev=stratum["kinetic_energy_ev"],
                        pitch_angle_rad=math.radians(stratum["pitch_angle_deg"]),
                        position_m=(fraction * wall_radius_m, 0.0, cell.launch_z_m),
                        parallel_direction=stratum["parallel_direction"],
                        gyrophase_rad=2.0 * math.pi * float(row[2]),
                        flux_surface_id=f"{cell_id}-{band['band_id']}",
                    )
                )
    if selected_keys is not None and len(launches) != len(selected_keys):
        raise ValueError("selected control keys were not all produced by the index ranges")
    return tuple(launches)


def candidate_records(*, namespace: str, design_key: str, cells: Sequence[LaunchCell], rule: Mapping[str, Any], wall_radius_m: float) -> list[dict[str, Any]]:
    """Campaign-free records of the complete candidate set (all stage-2 indices of every cell)."""

    total = int(rule["stage2_points_per_stratum"])
    launches = build_launches(
        "candidate",
        namespace=namespace,
        design_key=design_key,
        cells=cells,
        rule=rule,
        wall_radius_m=wall_radius_m,
        index_ranges={cell.cell_id: (0, total) for cell in cells},
    )
    return [
        {
            "key": key_of_launch(item),
            "kinetic_energy_ev": item.kinetic_energy_ev,
            "pitch_angle_rad": item.pitch_angle_rad,
            "position_m": list(item.position_m),
            "parallel_direction": item.parallel_direction,
            "gyrophase_rad": item.gyrophase_rad,
            "flux_surface_id": item.flux_surface_id,
        }
        for item in launches
    ]


def candidate_sha256(**kwargs: Any) -> str:
    return content_hash(candidate_records(**kwargs))


# --------------------------------------------------------------------------
# Estimators, allocation rule, control subset
# --------------------------------------------------------------------------


def wilson_width(successes: int, trials: int) -> float:
    estimate = wilson_interval(int(successes), int(trials))
    return estimate.upper - estimate.lower


def binomial_floor(successes: int, trials: int) -> float:
    """Standard error of the point estimate, sqrt(p(1-p)/n) (zero at p in {0, 1})."""

    p = successes / trials
    return math.sqrt(p * (1.0 - p) / trials)


def jeffreys_floor(successes: int, trials: int) -> float:
    """Standard error at the Jeffreys posterior mean (k+1/2)/(n+1); never zero."""

    p = (successes + 0.5) / (trials + 1.0)
    return math.sqrt(p * (1.0 - p) / trials)


def allocation_decision(stage1_counts: Mapping[str, Mapping[str, int]], rule: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen two-stage rule: top a cell up iff its stage-1 Wilson width exceeds the threshold.

    ``stage1_counts[cell_id] = {"wall_hit": k, "trials": n}``. Pure function; the worker
    applies it and the main process replays it from the endpoint terminations.
    """

    threshold = float(rule["wilson_width_threshold"])
    n1 = int(rule["stage1_launches_per_cell"])
    cells = {}
    for cell_id in sorted(stage1_counts):
        counts = stage1_counts[cell_id]
        k = int(counts["wall_hit"])
        n = int(counts["trials"])
        if n != n1:
            raise ValueError(f"{cell_id}: stage-1 trial count {n} differs from the rule's {n1}")
        width = wilson_width(k, n)
        cells[cell_id] = {
            "stage1_wall_hit": k,
            "stage1_trials": n,
            "stage1_wilson_width": width,
            "threshold": threshold,
            "topped_up": bool(width > threshold),
            "saturated": bool(width <= threshold),
        }
    topped = sorted(cell_id for cell_id, item in cells.items() if item["topped_up"])
    return {
        "rule": rule["statement"],
        "wilson_width_threshold": threshold,
        "cells": cells,
        "topped_up_cell_ids": topped,
        "topped_up_cell_count": len(topped),
        "saturated_cell_count": len(cells) - len(topped),
        "stage2_index_range": [int(rule["stage1_points_per_stratum"]), int(rule["stage2_points_per_stratum"])],
        "stage2_launch_count": len(topped) * (int(rule["stage2_points_per_stratum"]) - int(rule["stage1_points_per_stratum"])) * int(rule["strata_per_cell"]),
    }


def control_selection(
    *, namespace: str, design_key: str, cell_keys: Mapping[str, Sequence[str]], fraction: float, rounding: str
) -> dict[str, list[str]]:
    """Frozen-seed random subset of every cell's final launch keys (``ceil(fraction * n)`` per cell)."""

    from .sobol import seed_from_bytes

    if rounding != "ceil":
        raise ValueError("unknown control rounding rule")
    output: dict[str, list[str]] = {}
    for cell_id in sorted(cell_keys):
        keys = sorted(cell_keys[cell_id])
        if len(set(keys)) != len(keys) or not keys:
            raise ValueError(f"{cell_id}: control candidates must be unique and non-empty")
        count = int(math.ceil(float(fraction) * len(keys)))
        rng = np.random.default_rng(seed_from_bytes(f"{namespace}:{design_key}:{cell_id}:control".encode("utf-8")))
        chosen = rng.choice(len(keys), size=count, replace=False)
        output[cell_id] = sorted(keys[int(index)] for index in chosen)
    return output


def cell_counts_from_terminations(terminations: Mapping[str, str], cells_of_design: Sequence[LaunchCell]) -> dict[str, dict[str, int]]:
    """Per-cell counts from ``{launch_key: termination}``."""

    names = {cell.index: cell.cell_id for cell in cells_of_design}
    cells: dict[str, dict[str, int]] = {}
    for key, termination in terminations.items():
        cell = cells.setdefault(names[key_cell_index(key)], {"trials": 0, "wall_hit": 0, "reflected": 0, "domain_escape": 0, "timeout": 0})
        cell["trials"] += 1
        if termination in ("wall_hit", "reflected", "domain_escape"):
            cell[termination] += 1
        else:
            cell["timeout"] += 1
    return dict(sorted(cells.items()))


def _estimate(successes: int, trials: int) -> dict[str, Any]:
    return asdict(wilson_interval(int(successes), int(trials)))


def pooled_cell_row(
    cell: LaunchCell,
    stage1: Mapping[str, int],
    stage2: Mapping[str, int] | None,
    decision: Mapping[str, Any],
    control: Mapping[str, Any] | None,
    *,
    readiness_floor: float,
) -> dict[str, Any]:
    """Per-cell dataset row: stage counts, pooled Wilson intervals, floors, control."""

    names = ("wall_hit", "reflected", "domain_escape", "timeout")
    pooled = {name: int(stage1[name]) + (0 if stage2 is None else int(stage2[name])) for name in names}
    n = int(stage1["trials"]) + (0 if stage2 is None else int(stage2["trials"]))
    if sum(pooled.values()) != n:
        raise ValueError(f"{cell.cell_id}: pooled counts do not partition the trials")
    floor_plain = binomial_floor(pooled["wall_hit"], n)
    floor_jeffreys = jeffreys_floor(pooled["wall_hit"], n)
    return {
        **cell.to_dict(),
        "stage1": {"trials": int(stage1["trials"]), **{name: int(stage1[name]) for name in names}, "wilson_width": decision["stage1_wilson_width"]},
        "topped_up": bool(decision["topped_up"]),
        "saturated_after_stage1": bool(decision["saturated"]),
        "stage2": None if stage2 is None else {"trials": int(stage2["trials"]), **{name: int(stage2[name]) for name in names}},
        "final": {
            "trials": n,
            **pooled,
            "p_wall": _estimate(pooled["wall_hit"], n),
            "p_reflected": _estimate(pooled["reflected"], n),
            "p_escape": _estimate(pooled["domain_escape"], n),
            "p_timeout": _estimate(pooled["timeout"], n),
            "wilson_width": wilson_width(pooled["wall_hit"], n),
            "binomial_floor": floor_plain,
            "jeffreys_floor": floor_jeffreys,
            "surrogate_ready": bool(floor_jeffreys <= readiness_floor),
            "readiness_floor": readiness_floor,
        },
        "stage2_only": None
        if stage2 is None
        else {"p_wall": _estimate(int(stage2["wall_hit"]), int(stage2["trials"]))},
        "control": control,
    }


def design_pooled(cell_rows: Sequence[Mapping[str, Any]], *, weight: str) -> dict[str, Any]:
    """Design average of the per-cell wall-hit estimates with declared weights.

    ``weight`` is ``"wall_area"`` (cell wall areas from the catalogue; the declared
    design value) or ``"launches"`` (final launch counts; equals the raw pooled binomial
    fraction and is the closest analogue of v1's pooled value). The interval is the
    normal-approximation interval of the weighted mean with per-cell binomial variances.
    """

    if weight == "wall_area":
        weights = [float(row["wall_area_m2"]) for row in cell_rows]
    elif weight == "launches":
        weights = [float(row["final"]["trials"]) for row in cell_rows]
    else:
        raise ValueError("unknown pooling weight")
    total = sum(weights)
    probabilities = [row["final"]["p_wall"]["probability"] for row in cell_rows]
    variances = [
        (row["final"]["p_wall"]["probability"] * (1.0 - row["final"]["p_wall"]["probability"]) / row["final"]["trials"])
        for row in cell_rows
    ]
    mean = sum(w * p for w, p in zip(weights, probabilities)) / total
    standard = math.sqrt(sum((w / total) ** 2 * v for w, v in zip(weights, variances)))
    return {
        "weight": weight,
        "weights": [w / total for w in weights],
        "probability": mean,
        "standard_uncertainty": standard,
        "lower": max(0.0, mean - WILSON_Z * standard),
        "upper": min(1.0, mean + WILSON_Z * standard),
        "interval_method": "normal-approximation of the weighted mean with per-cell binomial variances (95 %)",
        "trials": sum(int(row["final"]["trials"]) for row in cell_rows),
    }
