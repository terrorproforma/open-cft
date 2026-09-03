"""Derived physical geometry and field features of every screening design.

Every feature is a DETERMINISTIC function of the committed screening dataset row
(``geometry``, ``field.sweep_qois`` and ``launch_design.cell_to_field`` of
``geometry-wall-loss-dataset.json``); nothing here is fitted, tuned or learned.
The dataset row itself is a committed artifact of
``orbit_wall_loss_geometry_screening_v1`` (bound by byte hash and Git blob), and
its geometry / field records are copied verbatim from the committed L1a geometry
sweep v2 results.  This module is bound by hash in the campaign's code contract.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from cft_revival.experiment_runtime.canonical import strict_json_file

from ..wall_loss_geometry_surrogate_v1 import data as v1d

STAGE_COUNTS = (3, 4, 5)

# name -> (unit, kind, provenance)
FEATURE_TABLE: tuple[tuple[str, str, str, str], ...] = (
    ("straight_length_m", "m", "continuous", "geometry.exit_start_m - geometry.injector_length_m (channel straight length L from the injector face to the exit-section start)"),
    ("wall_radius_m", "m", "continuous", "geometry.wall_radius_m (r_w of the straight dielectric channel)"),
    ("exit_start_m", "m", "continuous", "geometry.exit_start_m (axial start of the exit section)"),
    ("exit_length_m", "m", "continuous", "geometry.exit_length_m (realised divergent exit-section length; 0 when the sweep suppressed the exit)"),
    ("exit_length_fraction_realised", "1", "continuous", "geometry.exit_length_m / geometry.chamber_length_m (realised, after the sweep's minimum-length rule; NOT the design selector)"),
    ("exit_outer_radius_ratio", "1", "continuous", "geometry.exit_outer_radius_m / geometry.wall_radius_m (1 when there is no divergent exit)"),
    ("stage_count", "1", "discrete", "geometry.stage_count (realised PPM stage count 3/4/5)"),
    ("stage_count_is_3", "1", "discrete", "1 if geometry.stage_count == 3 else 0"),
    ("stage_count_is_4", "1", "discrete", "1 if geometry.stage_count == 4 else 0"),
    ("stage_count_is_5", "1", "discrete", "1 if geometry.stage_count == 5 else 0"),
    ("stage_pitch_m", "m", "continuous", "geometry.stage_pitch_m (axial magnet period)"),
    ("first_polarity", "1", "discrete", "geometry.first_polarity (+1 / -1 realised polarity of the first stage)"),
    ("magnet_axial_thickness_m", "m", "continuous", "geometry.magnet_axial_thickness_m (realised axial magnet thickness)"),
    ("magnet_inner_radius_m", "m", "continuous", "geometry.magnet_inner_radius_m"),
    ("magnet_radial_thickness_m", "m", "continuous", "geometry.magnet_outer_radius_m - geometry.magnet_inner_radius_m"),
    ("bore_max_b_t", "T", "continuous", "field.bore_max_b_t (maximum |B| on the screening's bore grid)"),
    ("centreline_abs_bz_peak_t", "T", "continuous", "field.sweep_qois.centreline_abs_bz_peak_t (L1a sweep QoI)"),
    ("centreline_mid_abs_bz_t", "T", "continuous", "field.sweep_qois.centreline_mid_abs_bz_t (L1a sweep QoI)"),
    ("log10_minimum_mirror_ratio", "1", "continuous", "log10(field.sweep_qois.minimum_mirror_ratio) (L1a sweep QoI; log because the ratios span two decades)"),
    ("log10_maximum_mirror_ratio", "1", "continuous", "log10(field.sweep_qois.maximum_mirror_ratio) (L1a sweep QoI)"),
    ("stage_gradient_rms_t_per_m", "T/m", "continuous", "field.sweep_qois.stage_gradient_rms_t_per_m (L1a sweep QoI)"),
    ("log10_field_energy_j", "1", "continuous", "log10(field.sweep_qois.field_energy_j) (L1a sweep QoI)"),
    ("boundary_to_peak_ratio", "1", "continuous", "field.sweep_qois.boundary_to_peak_ratio (L1a sweep QoI)"),
    ("cell1_cusp_distance_pitches", "1", "continuous", "launch_design.cell_to_field[gs1-cell-1].nearest_axis_bz_peak_distance_m / geometry.stage_pitch_m (axial distance from the cell centre to the nearest axis |Bz| peak, in pitches)"),
    ("cell1_null_distance_pitches", "1", "continuous", "launch_design.cell_to_field[gs1-cell-1].nearest_axis_null_distance_m / geometry.stage_pitch_m (distance to the nearest axis Bz null / cusp plane, in pitches)"),
    ("cell2_cusp_distance_pitches", "1", "continuous", "launch_design.cell_to_field[gs1-cell-2].nearest_axis_bz_peak_distance_m / geometry.stage_pitch_m"),
    ("cell2_null_distance_pitches", "1", "continuous", "launch_design.cell_to_field[gs1-cell-2].nearest_axis_null_distance_m / geometry.stage_pitch_m"),
    ("cell3_cusp_distance_pitches", "1", "continuous", "launch_design.cell_to_field[gs1-cell-3].nearest_axis_bz_peak_distance_m / geometry.stage_pitch_m"),
    ("cell3_null_distance_pitches", "1", "continuous", "launch_design.cell_to_field[gs1-cell-3].nearest_axis_null_distance_m / geometry.stage_pitch_m"),
    ("cell4_cusp_distance_pitches", "1", "continuous", "launch_design.cell_to_field[gs1-cell-4].nearest_axis_bz_peak_distance_m / geometry.stage_pitch_m"),
    ("cell4_null_distance_pitches", "1", "continuous", "launch_design.cell_to_field[gs1-cell-4].nearest_axis_null_distance_m / geometry.stage_pitch_m"),
)
FEATURE_NAMES: tuple[str, ...] = tuple(item[0] for item in FEATURE_TABLE)
FEATURE_UNITS: tuple[str, ...] = tuple(item[1] for item in FEATURE_TABLE)
FEATURE_KINDS: dict[str, str] = {item[0]: item[2] for item in FEATURE_TABLE}
FEATURE_PROVENANCE: dict[str, str] = {item[0]: item[3] for item in FEATURE_TABLE}
DISCRETE_FEATURES: tuple[str, ...] = tuple(name for name, kind in FEATURE_KINDS.items() if kind == "discrete")
CONTINUOUS_FEATURES: tuple[str, ...] = tuple(name for name, kind in FEATURE_KINDS.items() if kind == "continuous")

EXCLUDED_RECORDED_QUANTITIES: dict[str, str] = {
    "geometry.chamber_length_m": "equals stage_count * stage_pitch_m exactly for every design (verified); kept only as the extrapolation-cluster rule",
    "field.sweep_qois.axis_cusp_positions_m (count)": "count equals stage_count for every design (verified); positions enter through the per-cell cusp distances",
    "field.sweep_qois.axis_null_positions_m (count)": "count equals stage_count + 1 for every design (verified); positions enter through the per-cell null distances",
    "field.sweep_qois.field_peak_t": "near-duplicate of bore_max_b_t (grid peak vs bore-grid peak); one peak descriptor is kept",
    "geometry.injector_length_m": "enters through straight_length_m",
    "geometry.exit_outer_radius_m": "enters through exit_outer_radius_ratio",
    "geometry.dielectric_thickness_m": "not a channel-interior quantity; wall_radius_m carries the interior geometry",
    "design_values.* (the eleven v1 raw design parameters)": "deliberately excluded: v1 showed the step discontinuities of the design -> geometry map dominate; v2 tests realised geometry instead",
}


def derive_features(record: Mapping[str, Any], cells: Sequence[str]) -> dict[str, float]:
    """Feature dictionary of one dataset row (deterministic; raises on any inconsistency)."""

    geometry = record["geometry"]
    field = record["field"]
    qois = field["sweep_qois"]
    stage_count = int(geometry["stage_count"])
    if stage_count not in STAGE_COUNTS:
        raise ValueError(f"{record['case_id']}: stage count {stage_count} outside the sweep's 3..5")
    pitch = float(geometry["stage_pitch_m"])
    chamber_length = float(geometry["chamber_length_m"])
    if pitch <= 0.0 or not math.isclose(chamber_length, stage_count * pitch, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{record['case_id']}: chamber length is not stage_count * pitch")
    straight = float(geometry["exit_start_m"]) - float(geometry["injector_length_m"])
    if straight <= 0.0:
        raise ValueError(f"{record['case_id']}: non-positive straight length")
    if len(qois["axis_cusp_positions_m"]) != stage_count or len(qois["axis_null_positions_m"]) != stage_count + 1:
        raise ValueError(f"{record['case_id']}: axis cusp/null counts differ from the stage rule")
    wall_radius = float(geometry["wall_radius_m"])
    by_cell = {item["cell_id"]: item for item in record["launch_design"]["cell_to_field"]}
    features: dict[str, float] = {
        "straight_length_m": straight,
        "wall_radius_m": wall_radius,
        "exit_start_m": float(geometry["exit_start_m"]),
        "exit_length_m": float(geometry["exit_length_m"]),
        "exit_length_fraction_realised": float(geometry["exit_length_m"]) / chamber_length,
        "exit_outer_radius_ratio": float(geometry["exit_outer_radius_m"]) / wall_radius,
        "stage_count": float(stage_count),
        "stage_count_is_3": 1.0 if stage_count == 3 else 0.0,
        "stage_count_is_4": 1.0 if stage_count == 4 else 0.0,
        "stage_count_is_5": 1.0 if stage_count == 5 else 0.0,
        "stage_pitch_m": pitch,
        "first_polarity": float(int(geometry["first_polarity"])),
        "magnet_axial_thickness_m": float(geometry["magnet_axial_thickness_m"]),
        "magnet_inner_radius_m": float(geometry["magnet_inner_radius_m"]),
        "magnet_radial_thickness_m": float(geometry["magnet_outer_radius_m"]) - float(geometry["magnet_inner_radius_m"]),
        "bore_max_b_t": float(field["bore_max_b_t"]),
        "centreline_abs_bz_peak_t": float(qois["centreline_abs_bz_peak_t"]),
        "centreline_mid_abs_bz_t": float(qois["centreline_mid_abs_bz_t"]),
        "log10_minimum_mirror_ratio": math.log10(float(qois["minimum_mirror_ratio"])),
        "log10_maximum_mirror_ratio": math.log10(float(qois["maximum_mirror_ratio"])),
        "stage_gradient_rms_t_per_m": float(qois["stage_gradient_rms_t_per_m"]),
        "log10_field_energy_j": math.log10(float(qois["field_energy_j"])),
        "boundary_to_peak_ratio": float(qois["boundary_to_peak_ratio"]),
    }
    for index, cell in enumerate(cells, start=1):
        item = by_cell[cell]
        features[f"cell{index}_cusp_distance_pitches"] = float(item["nearest_axis_bz_peak_distance_m"]) / pitch
        features[f"cell{index}_null_distance_pitches"] = float(item["nearest_axis_null_distance_m"]) / pitch
    if features["first_polarity"] not in (-1.0, 1.0):
        raise ValueError(f"{record['case_id']}: polarity is not +/-1")
    if set(features) != set(FEATURE_NAMES):
        raise ValueError("feature table and derivation disagree")
    for name, value in features.items():
        if not math.isfinite(value):
            raise ValueError(f"{record['case_id']}: non-finite feature {name}")
    return features


def feature_vector(features: Mapping[str, float]) -> tuple[float, ...]:
    return tuple(float(features[name]) for name in FEATURE_NAMES)


def load_feature_rows(spec: Mapping[str, Any], output_spec: Mapping[str, Any]) -> tuple[v1d.DesignRow, ...]:
    """v1's validated rows (labels, batch, chamber length) with ``inputs`` replaced by the derived features."""

    base = v1d.load_rows(spec, (), output_spec)
    dataset = strict_json_file(v1d.REPOSITORY / spec["dataset_path"])
    records = {record["case_id"]: record for record in dataset["designs"]}
    cells = tuple(output_spec["cells"])
    rows = []
    for row in base:
        features = derive_features(records[row.case_id], cells)
        if int(features["stage_count"]) != row.stage_count:
            raise ValueError(f"{row.case_id}: stage count disagreement")
        rows.append(replace(row, inputs=feature_vector(features)))
    return tuple(rows)


def feature_degeneracy_report(rows: Sequence[v1d.DesignRow], *, minimum_distinct: int = 8, minimum_level_count: int = 5) -> dict[str, Any]:
    """Continuous features need >= 8 distinct values; discrete ones >= 2 levels each with >= 5 designs."""

    distinct: dict[str, int] = {}
    level_counts: dict[str, dict[str, int]] = {}
    failures = []
    for index, name in enumerate(FEATURE_NAMES):
        values = [row.inputs[index] for row in rows]
        distinct[name] = len(set(values))
        if FEATURE_KINDS[name] == "continuous":
            if distinct[name] < minimum_distinct:
                failures.append(name)
        else:
            counts: dict[str, int] = {}
            for value in values:
                counts[str(value)] = counts.get(str(value), 0) + 1
            level_counts[name] = counts
            if len(counts) < 2 or min(counts.values()) < minimum_level_count:
                failures.append(name)
    tuples = {row.inputs for row in rows}
    if len(tuples) != len(rows):
        failures.append("<duplicate feature tuples>")
    return {
        "distinct_values_per_feature": distinct,
        "discrete_level_counts": level_counts,
        "distinct_feature_tuples": len(tuples),
        "rows": len(rows),
        "minimum_distinct_required_continuous": minimum_distinct,
        "minimum_level_count_required_discrete": minimum_level_count,
        "failures": failures,
        "passed": not failures,
    }


def feature_manifest() -> dict[str, Any]:
    return {
        "derived_not_fitted": True,
        "statement": "every feature is a deterministic arithmetic function of the committed screening dataset row (geometry, field.sweep_qois, launch_design.cell_to_field); no feature is fitted, tuned, scaled by data statistics, or learned; the extraction code (features.py) is bound by hash in the code contract",
        "names": list(FEATURE_NAMES),
        "units": list(FEATURE_UNITS),
        "kinds": dict(FEATURE_KINDS),
        "provenance": dict(FEATURE_PROVENANCE),
        "excluded_recorded_quantities": dict(EXCLUDED_RECORDED_QUANTITIES),
        "count": len(FEATURE_NAMES),
    }
