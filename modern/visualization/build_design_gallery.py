"""Build a deterministic gallery of representative L0 operating points.

The gallery is reconstructed from the checked sweep configuration through the
production L0 sampling and Python-reference physics APIs.  It contains no
timestamps or measured runtime, so identical code and configuration inputs
produce byte-identical JSON.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from cft_revival.physics.reference import evaluate_batch
from cft_revival.physics.workflows import (
    L0_MODEL_CLAIM,
    L0_MODEL_FIDELITY,
    load_l0_json,
    operating_point_to_dict,
    result_to_dict,
    sweep_points_from_config,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "l0-deterministic-sweep.json"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("design-gallery.json")

OBJECTIVES: tuple[dict[str, str], ...] = (
    {
        "path": "result.axial_thrust_n",
        "direction": "maximize",
        "unit": "N",
    },
    {
        "path": "result.specific_impulse_s",
        "direction": "maximize",
        "unit": "s",
    },
    {
        "path": "result.power_budget.ppu_input_to_beam_efficiency",
        "direction": "maximize",
        "unit": "1",
    },
    {
        "path": "result.power_budget.anode_input_power_w",
        "direction": "minimize",
        "unit": "W",
    },
)

COMMON_CAVEATS = [
    (
        "This is a hypothetical L0 operating concept from a 0D/global "
        "conservation-reduced performance model, not validated hardware or geometry."
    ),
    (
        "Charge-state, utilization, beam-current, divergence, cathode-power, "
        "and PPU-boundary factors are supplied inputs rather than plasma closures."
    ),
    (
        "Numerical conservation closure does not establish experimental "
        "calibration, predictive accuracy, feasibility, lifetime, or manufacturability."
    ),
]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _field(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    return value


def _finite_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"objective value must be a real number or null, got {value!r}")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"objective value must be finite or null, got {value!r}")
    return converted


def _dataset_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the canonical ordered list of complete indexed sweep records."""

    digest = sha256()
    digest.update(b"[")
    for position, record in enumerate(records):
        if position:
            digest.update(b",")
        digest.update(_canonical_json(record))
    digest.update(b"]")
    return digest.hexdigest()


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot calculate a median for an empty sequence")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _best(
    records: Sequence[Mapping[str, Any]],
    eligible_indices: Sequence[int],
    *,
    path: str,
    direction: str,
) -> tuple[int, float, int]:
    candidates: list[tuple[int, float]] = []
    for index in eligible_indices:
        value = _finite_number(_field(records[index], path))
        if value is not None:
            candidates.append((index, value))
    if not candidates:
        raise ValueError(f"selection {path!r} has no finite eligible values")
    if direction == "maximize":
        optimum = max(value for _index, value in candidates)
    elif direction == "minimize":
        optimum = min(value for _index, value in candidates)
    else:
        raise ValueError(f"unsupported objective direction {direction!r}")
    tied = [index for index, value in candidates if value == optimum]
    return min(tied), optimum, len(tied)


def _normalization(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for objective in OBJECTIVES:
        path = objective["path"]
        values = [
            value
            for record in records
            if (value := _finite_number(_field(record, path))) is not None
        ]
        if not values or min(values) == max(values):
            raise ValueError(f"objective {path!r} has no finite non-zero range")
        definitions[path] = {
            "direction": objective["direction"],
            "unit": objective["unit"],
            "minimum": min(values),
            "maximum": max(values),
            "weight": 1.0 / len(OBJECTIVES),
            "formula": (
                "(value - minimum) / (maximum - minimum)"
                if objective["direction"] == "maximize"
                else "(maximum - value) / (maximum - minimum)"
            ),
        }
    return definitions


def _compromise(
    records: Sequence[Mapping[str, Any]],
    normalization: Mapping[str, Mapping[str, Any]],
) -> tuple[int, float, int, dict[int, float]]:
    scores: dict[int, float] = {}
    for index, record in enumerate(records):
        terms: list[float] = []
        for path, definition in normalization.items():
            value = _finite_number(_field(record, path))
            if value is None:
                break
            minimum = float(definition["minimum"])
            maximum = float(definition["maximum"])
            normalized = (
                (value - minimum) / (maximum - minimum)
                if definition["direction"] == "maximize"
                else (maximum - value) / (maximum - minimum)
            )
            terms.append(float(definition["weight"]) * normalized)
        if len(terms) == len(normalization):
            scores[index] = sum(terms)
    if not scores:
        raise ValueError("normalized compromise has no complete finite candidates")
    optimum = max(scores.values())
    tied = [index for index, score in scores.items() if score == optimum]
    return min(tied), optimum, len(tied), scores


def _sampled_inputs(record: Mapping[str, Any]) -> dict[str, float]:
    point = record["input"]
    fractions = point["charge_state_number_fractions"]
    utilization = float(point["mass_utilization_fraction_of_inlet_mass"])
    thruster_power = float(
        record["result"]["power_budget"]["thruster_electrical_input_power_w"]
    )
    requested_ppu_power = float(point["ppu_input_power_w"])
    return {
        "discharge_voltage_v": float(point["discharge_voltage_v"]),
        "propellant_mass_flow_kg_per_s": float(
            point["propellant_mass_flow_kg_per_s"]
        ),
        "ionized_number_fraction": utilization,
        "xe_double_plus_fraction_of_ions": (
            float(fractions["xe_double_plus"]) / utilization
        ),
        "beam_current_fraction_of_anode_current": float(
            point["beam_current_fraction_of_anode_current"]
        ),
        "axial_momentum_fraction_of_ion_momentum": float(
            point["axial_momentum_fraction_of_ion_momentum"]
        ),
        "cathode_input_power_w": float(point["cathode_input_power_w"]),
        "ppu_efficiency_fraction": thruster_power / requested_ppu_power,
    }


def _selection(
    *,
    rule: str,
    objective_path: str,
    direction: str,
    optimum: float,
    eligible_count: int,
    finite_candidate_count: int,
    tied_best_count: int,
    constraint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "rule": rule,
        "objective_path": objective_path,
        "direction": direction,
        "objective_value": optimum,
        "constraint": dict(constraint) if constraint is not None else None,
        "eligible_count": eligible_count,
        "finite_candidate_count": finite_candidate_count,
        "rank": 1,
        "rank_definition": (
            "Competition rank among eligible finite candidates in the stated "
            "objective direction."
        ),
        "exact_tie_count_at_rank_1": tied_best_count,
        "tie_definition": "Exact binary64 equality of the ranking value.",
        "tie_break": "Select the lowest zero-based deterministic sweep index.",
    }


def _concept(
    records: Sequence[Mapping[str, Any]],
    *,
    concept_id: str,
    label: str,
    index: int,
    selection: Mapping[str, Any],
    extra_caveat: str,
    score_components: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    source = records[index]
    return {
        "concept_id": concept_id,
        "entry_type": "representative operating point",
        "label": label,
        "index": index,
        "sampled_inputs": _sampled_inputs(source),
        "input": deepcopy(source["input"]),
        "result": deepcopy(source["result"]),
        "selection": dict(selection),
        "normalized_score_components": (
            dict(score_components) if score_components is not None else None
        ),
        "caveats": [*COMMON_CAVEATS, extra_caveat],
    }


def _score_components(
    record: Mapping[str, Any],
    normalization: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    components: dict[str, float] = {}
    for path, definition in normalization.items():
        value = _finite_number(_field(record, path))
        if value is None:
            raise ValueError("selected compromise point has a null objective")
        minimum = float(definition["minimum"])
        maximum = float(definition["maximum"])
        components[path] = (
            (value - minimum) / (maximum - minimum)
            if definition["direction"] == "maximize"
            else (maximum - value) / (maximum - minimum)
        )
    return components


def _residual_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    paths = {
        "maximum_absolute_particle_rate_residual_particles_per_s": (
            "result.diagnostics.particle_rate_residual_particles_per_s"
        ),
        "maximum_absolute_mass_flow_residual_kg_per_s": (
            "result.diagnostics.mass_flow_residual_kg_per_s"
        ),
        "maximum_absolute_beam_current_residual_a": (
            "result.diagnostics.beam_current_residual_a"
        ),
        "maximum_absolute_beam_power_residual_w": (
            "result.diagnostics.beam_power_residual_w"
        ),
    }
    return {
        name: max(abs(float(_field(record, path))) for record in records)
        for name, path in paths.items()
    }


def build_gallery(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Reproduce the sweep and select five transparent representative points."""

    config_path = config_path.resolve()
    config = dict(load_l0_json(config_path))
    points, sampling = sweep_points_from_config(config)
    results = evaluate_batch(points)
    records = [
        {
            "index": index,
            "input": operating_point_to_dict(point),
            "result": result_to_dict(result),
        }
        for index, (point, result) in enumerate(zip(points, results, strict=True))
    ]
    all_indices = list(range(len(records)))
    thrust_values = [
        float(_field(record, "result.axial_thrust_n")) for record in records
    ]
    thrust_minimum = min(thrust_values)
    thrust_maximum = max(thrust_values)
    useful_thrust = _median(thrust_values)
    useful_indices = [
        index
        for index, value in enumerate(thrust_values)
        if value >= useful_thrust
    ]
    threshold_constraint = {
        "path": "result.axial_thrust_n",
        "operator": ">=",
        "threshold": useful_thrust,
        "unit": "N",
        "derivation": (
            "Median of all 8,192 finite axial-thrust values; for this even-sized "
            "dataset, the arithmetic mean of sorted positions 4,095 and 4,096."
        ),
        "rationale": (
            "A dataset-relative, visible threshold retains the upper half of the "
            "sampled thrust range before comparing power or efficiency."
        ),
    }

    max_thrust_index, max_thrust, max_thrust_ties = _best(
        records,
        all_indices,
        path="result.axial_thrust_n",
        direction="maximize",
    )
    max_isp_index, max_isp, max_isp_ties = _best(
        records,
        all_indices,
        path="result.specific_impulse_s",
        direction="maximize",
    )
    min_power_index, min_power, min_power_ties = _best(
        records,
        useful_indices,
        path="result.power_budget.anode_input_power_w",
        direction="minimize",
    )
    best_ppu_index, best_ppu, best_ppu_ties = _best(
        records,
        useful_indices,
        path="result.power_budget.ppu_input_to_beam_efficiency",
        direction="maximize",
    )
    normalization = _normalization(records)
    compromise_index, compromise_score, compromise_ties, scores = _compromise(
        records, normalization
    )

    concepts = [
        _concept(
            records,
            concept_id="maximum-axial-thrust",
            label="Maximum axial thrust L0 operating concept",
            index=max_thrust_index,
            selection=_selection(
                rule="Maximize axial thrust over all sweep points.",
                objective_path="result.axial_thrust_n",
                direction="maximize",
                optimum=max_thrust,
                eligible_count=len(records),
                finite_candidate_count=len(records),
                tied_best_count=max_thrust_ties,
            ),
            extra_caveat=(
                "Maximum thrust is only relative to this sampled L0 domain and "
                "does not imply a feasible physical thruster."
            ),
        ),
        _concept(
            records,
            concept_id="maximum-specific-impulse",
            label="Maximum specific impulse L0 operating concept",
            index=max_isp_index,
            selection=_selection(
                rule="Maximize specific impulse over all sweep points.",
                objective_path="result.specific_impulse_s",
                direction="maximize",
                optimum=max_isp,
                eligible_count=len(records),
                finite_candidate_count=len(records),
                tied_best_count=max_isp_ties,
            ),
            extra_caveat=(
                "Maximum specific impulse is relative to the sampled L0 domain "
                "and does not account for discharge, wall, or lifetime constraints."
            ),
        ),
        _concept(
            records,
            concept_id="minimum-anode-power-useful-thrust",
            label="Minimum anode-input power useful-thrust L0 operating concept",
            index=min_power_index,
            selection=_selection(
                rule=(
                    "Among points at or above the dataset-median axial-thrust "
                    "threshold, minimize anode input power."
                ),
                objective_path="result.power_budget.anode_input_power_w",
                direction="minimize",
                optimum=min_power,
                eligible_count=len(useful_indices),
                finite_candidate_count=len(useful_indices),
                tied_best_count=min_power_ties,
                constraint=threshold_constraint,
            ),
            extra_caveat=(
                "The ranking minimizes anode input power; beam kinetic power is "
                "reported in the complete power budget but is not a second ranking key."
            ),
        ),
        _concept(
            records,
            concept_id="best-ppu-efficiency-useful-thrust",
            label="Best PPU-input-to-beam efficiency useful-thrust L0 operating concept",
            index=best_ppu_index,
            selection=_selection(
                rule=(
                    "Among points at or above the dataset-median axial-thrust "
                    "threshold, maximize PPU-input-to-beam efficiency."
                ),
                objective_path=(
                    "result.power_budget.ppu_input_to_beam_efficiency"
                ),
                direction="maximize",
                optimum=best_ppu,
                eligible_count=len(useful_indices),
                finite_candidate_count=len(useful_indices),
                tied_best_count=best_ppu_ties,
                constraint=threshold_constraint,
            ),
            extra_caveat=(
                "This efficiency uses the reported L0 PPU input boundary and is "
                "not hardware-qualified PPU efficiency."
            ),
        ),
        _concept(
            records,
            concept_id="normalized-equal-weight-compromise",
            label="Normalized equal-weight compromise representative operating point",
            index=compromise_index,
            selection=_selection(
                rule=(
                    "Maximize the arithmetic sum of four equal-weight min-max "
                    "desirabilities over complete finite sweep points."
                ),
                objective_path="selection.normalized_equal_weight_score",
                direction="maximize",
                optimum=compromise_score,
                eligible_count=len(records),
                finite_candidate_count=len(scores),
                tied_best_count=compromise_ties,
            ),
            score_components=_score_components(
                records[compromise_index], normalization
            ),
            extra_caveat=(
                "This is an explicitly weighted visualization heuristic, not a "
                "Pareto set, knee point, campaign-objective result, or physical optimum."
            ),
        ),
    ]

    try:
        config_reference = config_path.relative_to(ROOT).as_posix()
    except ValueError:
        config_reference = config_path.name
    gallery = {
        "document_type": (
            "cft-revival-l0-representative-operating-point-gallery"
        ),
        "schema_version": "1.0",
        "title": "Representative L0 operating-point gallery",
        "model": {
            "fidelity": L0_MODEL_FIDELITY,
            "claim": L0_MODEL_CLAIM,
            "dimensionality": "0D/global reduced performance",
            "hypothetical_inputs": True,
            "interpretation": (
                "Entries are L0 operating concepts or representative operating "
                "points, never validated physical thruster geometries."
            ),
            "not_solved_at_l0": [
                "No spatially resolved 1D, 2D, or 3D field solution exists.",
                "No magnet radii, channel dimensions, or validated geometry are evaluated.",
                "No magnetic-field topology or plasma-discharge solution exists.",
            ],
            "future_l1_candidate_fields": {
                "status": (
                    "Required future evidence before an L0 operating point can "
                    "become a physical design candidate."
                ),
                "fields": [
                    "inner and outer magnet radii and complete magnet/coil geometry",
                    "channel, shield, enclosure, and electrode geometry",
                    "spatial magnetic-field magnitude, direction, gradients, and topology",
                    "spatial plasma/discharge state and transport closure",
                    "wall interaction, thermal, erosion, lifetime, and material fields",
                    "facility and electrical boundary conditions",
                    "calibration, uncertainty, and experimental validation evidence",
                ],
            },
        },
        "source": {
            "config_path": config_reference,
            "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
            "config_hash_scope": "Exact source-file bytes.",
            "config": config,
            "sampling": sampling,
            "sample_count": len(records),
            "dataset_identity": {
                "algorithm": "sha256",
                "serialization": (
                    "UTF-8 canonical JSON (sorted keys, compact separators, "
                    "ASCII escapes, no NaN/Infinity) of the ordered complete "
                    "indexed input/result record list."
                ),
                "sha256": _dataset_sha256(records),
            },
            "evaluation_api": (
                "cft_revival.physics.reference.evaluate_batch over "
                "cft_revival.physics.workflows.sweep_points_from_config"
            ),
        },
        "selection_policy": {
            "useful_thrust_threshold": threshold_constraint,
            "useful_thrust_eligible_count": len(useful_indices),
            "observed_axial_thrust_range_n": {
                "minimum": thrust_minimum,
                "maximum": thrust_maximum,
            },
            "null_policy": (
                "Null objective values are excluded from that rule; non-finite "
                "values are rejected. Selected records retain documented nullable "
                "result fields as JSON null."
            ),
            "rank_and_tie_policy": (
                "Rank only eligible finite values in the stated direction; exact "
                "binary64 ties share competition rank and the lowest sweep index "
                "is the deterministic representative."
            ),
            "normalized_compromise": {
                "name": "equal-weight min-max desirability",
                "aggregation": (
                    "Sum weight * normalized desirability; all four weights are 0.25."
                ),
                "objectives": normalization,
                "classification": (
                    "Visualization heuristic only; not Pareto, knee, or campaign semantics."
                ),
            },
        },
        "conservation": {
            "all_sweep_point_maximum_absolute_residuals": _residual_summary(records),
            "selected_point_test_bounds": {
                "particle_rate": (
                    "abs(residual) <= 4e-16 * max(1, total particle rate)"
                ),
                "mass_flow": "abs(residual) <= 3e-21 kg/s",
                "beam_current": "abs(residual) <= 2e-13 A",
                "beam_power": (
                    "abs(residual) <= 5e-14 * max(1, beam kinetic power)"
                ),
                "fractions_and_efficiencies": (
                    "charge-state sum within 2e-15; utilization and efficiencies in [0, 1]"
                ),
            },
        },
        "concepts": concepts,
    }
    validate_gallery(gallery)
    return gallery


def validate_gallery(gallery: Mapping[str, Any]) -> None:
    """Reject wrong-shape, non-finite, or semantically mislabeled galleries."""

    if gallery.get("document_type") != (
        "cft-revival-l0-representative-operating-point-gallery"
    ):
        raise ValueError("unexpected gallery document_type")
    source = gallery.get("source")
    if not isinstance(source, Mapping) or source.get("sample_count") != 8192:
        raise ValueError("gallery must identify the 8,192-point source sweep")
    concepts = gallery.get("concepts")
    if not isinstance(concepts, list) or len(concepts) != 5:
        raise ValueError("gallery must contain exactly five representative points")
    indices: set[int] = set()
    for concept in concepts:
        if not isinstance(concept, Mapping):
            raise ValueError("gallery concepts must be objects")
        if concept.get("entry_type") != "representative operating point":
            raise ValueError("gallery entry type overclaims L0 semantics")
        index = concept.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 8192:
            raise ValueError("gallery concept index is outside the source sweep")
        if index in indices:
            raise ValueError("gallery concepts must select distinct sweep indices")
        indices.add(index)
        if not concept.get("caveats"):
            raise ValueError("every gallery concept requires caveats")

    def walk(value: object, path: str = "$") -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not isfinite(value):
                raise ValueError(f"non-finite gallery value at {path}")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"non-text gallery key at {path}")
                walk(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        raise ValueError(f"unsupported gallery value at {path}: {type(value).__name__}")

    walk(gallery)


def render_gallery(gallery: Mapping[str, Any]) -> str:
    validate_gallery(gallery)
    return json.dumps(
        gallery,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def generate(
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    gallery = build_gallery(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_gallery(gallery),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = generate(args.config.resolve(), args.output.resolve())
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
