"""Deterministic selection and semantic tests for the L0 point gallery."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from cft_revival.physics.reference import evaluate_batch
from cft_revival.physics.workflows import (
    load_l0_json,
    operating_point_to_dict,
    result_to_dict,
    sweep_points_from_config,
)

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "build_design_gallery.py"
GALLERY_PATH = MODERN / "visualization" / "design-gallery.json"
CONFIG_PATH = MODERN / "config" / "l0-deterministic-sweep.json"

# Provenance pin: SHA-256 of the exact committed (LF) bytes of
# config/l0-deterministic-sweep.json.  The previous value
# a4703ac1541539829f47f909d24d01d4996ed1da97a9d86e9e2323e54039fbbf was the
# hash of the same file smudged to CRLF by core.autocrlf=true before
# .gitattributes pinned eol=lf (fab0eccc); the dataset identity below is
# EOL-independent and unchanged.
EXPECTED_CONFIG_SHA256 = (
    "2d727b1af7d9be9f35f227cc318beae29af6cbd2fbead28842a4c17d67551b6b"
)
EXPECTED_DATASET_SHA256 = (
    "c0a36ed83655d8bef0e8419a27dfbc330926716dadb6c893b6ef6f9b2ddbae84"
)
EXPECTED_INDICES = {
    "maximum-axial-thrust": 6352,
    "maximum-specific-impulse": 2752,
    "minimum-anode-power-useful-thrust": 1633,
    "best-ppu-efficiency-useful-thrust": 148,
    "normalized-equal-weight-compromise": 1192,
}


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "design_gallery_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def source_data():
    config = dict(load_l0_json(CONFIG_PATH))
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
    return config, sampling, points, results, records


@pytest.fixture(scope="module")
def gallery():
    return GENERATOR.build_gallery()


def _concepts_by_id(gallery):
    return {concept["concept_id"]: concept for concept in gallery["concepts"]}


def test_checked_artifact_is_byte_deterministic_and_current(gallery, tmp_path: Path) -> None:
    generated = GENERATOR.render_gallery(gallery)
    assert GALLERY_PATH.read_text(encoding="utf-8") == generated
    assert generated == GENERATOR.render_gallery(GENERATOR.build_gallery())

    output = tmp_path / "gallery.json"
    GENERATOR.generate(output_path=output)
    assert output.read_text(encoding="utf-8") == generated
    assert json.loads(generated) == gallery


def test_exact_sweep_config_sampling_and_dataset_identity(gallery, source_data) -> None:
    config, sampling, _points, _results, records = source_data
    source = gallery["source"]
    assert source["sample_count"] == 8192
    assert source["config_path"] == "config/l0-deterministic-sweep.json"
    assert source["config"] == config
    assert source["sampling"] == sampling == {
        "method": "deterministic-prime-base-radical-inverse",
        "seed": 20260901,
        "batch_size": 8192,
        "dimensions": [
            "discharge_voltage_v",
            "propellant_mass_flow_kg_per_s",
            "ionized_number_fraction",
            "xe_double_plus_fraction_of_ions",
            "beam_current_fraction_of_anode_current",
            "axial_momentum_fraction_of_ion_momentum",
            "cathode_input_power_w",
            "ppu_efficiency_fraction",
        ],
    }
    assert source["config_sha256"] == EXPECTED_CONFIG_SHA256
    assert source["dataset_identity"]["sha256"] == EXPECTED_DATASET_SHA256
    assert GENERATOR._dataset_sha256(records) == EXPECTED_DATASET_SHA256


def test_useful_thrust_threshold_is_visible_dataset_median(gallery, source_data) -> None:
    _config, _sampling, _points, results, _records = source_data
    ordered = sorted(result.axial_thrust_n for result in results)
    expected = (ordered[4095] + ordered[4096]) / 2.0
    policy = gallery["selection_policy"]
    threshold = policy["useful_thrust_threshold"]
    assert threshold["threshold"] == expected == 0.016422927461904484
    assert threshold["operator"] == ">="
    assert policy["useful_thrust_eligible_count"] == 4096
    assert sum(result.axial_thrust_n >= expected for result in results) == 4096
    assert "Median" in threshold["derivation"]
    assert "upper half" in threshold["rationale"]


def test_selected_indices_and_extreme_rules_are_independently_reproduced(
    gallery, source_data
) -> None:
    _config, _sampling, _points, results, _records = source_data
    concepts = _concepts_by_id(gallery)
    assert {name: item["index"] for name, item in concepts.items()} == EXPECTED_INDICES

    threshold = gallery["selection_policy"]["useful_thrust_threshold"]["threshold"]
    useful = [
        index
        for index, result in enumerate(results)
        if result.axial_thrust_n >= threshold
    ]
    expected = {
        "maximum-axial-thrust": max(
            range(len(results)),
            key=lambda index: (results[index].axial_thrust_n, -index),
        ),
        "maximum-specific-impulse": max(
            range(len(results)),
            key=lambda index: (results[index].specific_impulse_s, -index),
        ),
        "minimum-anode-power-useful-thrust": min(
            useful,
            key=lambda index: (
                results[index].power_budget.anode_input_power_w,
                index,
            ),
        ),
        "best-ppu-efficiency-useful-thrust": max(
            useful,
            key=lambda index: (
                results[index].power_budget.ppu_input_to_beam_efficiency,
                -index,
            ),
        ),
    }
    for concept_id, index in expected.items():
        concept = concepts[concept_id]
        assert concept["index"] == index
        assert concept["selection"]["rank"] == 1
        assert concept["selection"]["exact_tie_count_at_rank_1"] == 1
        assert concept["selection"]["tie_break"].startswith("Select the lowest")


def test_normalized_compromise_rule_is_explicit_and_reproduced(
    gallery, source_data
) -> None:
    _config, _sampling, _points, _results, records = source_data
    policy = gallery["selection_policy"]["normalized_compromise"]
    definitions = policy["objectives"]
    assert policy["classification"].startswith("Visualization heuristic")
    assert {definition["direction"] for definition in definitions.values()} == {
        "maximize",
        "minimize",
    }
    assert all(definition["weight"] == 0.25 for definition in definitions.values())

    scores = {}
    for index, record in enumerate(records):
        score = 0.0
        for path, definition in definitions.items():
            value = float(GENERATOR._field(record, path))
            minimum = definition["minimum"]
            maximum = definition["maximum"]
            desirability = (
                (value - minimum) / (maximum - minimum)
                if definition["direction"] == "maximize"
                else (maximum - value) / (maximum - minimum)
            )
            score += definition["weight"] * desirability
        scores[index] = score

    expected_index = max(scores, key=lambda index: (scores[index], -index))
    concept = _concepts_by_id(gallery)["normalized-equal-weight-compromise"]
    assert expected_index == concept["index"] == 1192
    assert scores[expected_index] == concept["selection"]["objective_value"]
    assert sum(
        0.25 * value
        for value in concept["normalized_score_components"].values()
    ) == concept["selection"]["objective_value"]


def test_entries_retain_every_sampled_input_and_complete_result(gallery, source_data) -> None:
    _config, sampling, _points, _results, records = source_data
    sampled_names = set(sampling["dimensions"])
    for concept in gallery["concepts"]:
        source = records[concept["index"]]
        assert concept["input"] == source["input"]
        assert concept["result"] == source["result"]
        assert set(concept["sampled_inputs"]) == sampled_names
        assert set(concept["input"]["charge_state_number_fractions"]) == {
            "xe_neutral",
            "xe_plus",
            "xe_double_plus",
        }
        assert "mass_utilization_fraction_of_inlet_mass" in concept["input"]
        assert "beam_current_fraction_of_anode_current" in concept["input"]
        assert "axial_momentum_fraction_of_ion_momentum" in concept["input"]
        assert set(concept["result"]) == {
            "total_xenon_particle_rate_per_s",
            "neutral_particle_rate_per_s",
            "xe_plus_particle_rate_per_s",
            "xe_double_plus_particle_rate_per_s",
            "xe_plus_speed_m_per_s",
            "xe_double_plus_speed_m_per_s",
            "undiverged_ion_thrust_n",
            "axial_thrust_n",
            "specific_impulse_s",
            "power_budget",
            "diagnostics",
            "applicability_warnings",
        }


def test_finite_and_documented_null_handling(gallery) -> None:
    GENERATOR.validate_gallery(gallery)

    nullable = deepcopy(gallery)
    nullable["concepts"][0]["result"]["power_budget"][
        "anode_to_beam_efficiency"
    ] = None
    GENERATOR.validate_gallery(nullable)
    assert '"anode_to_beam_efficiency": null' in GENERATOR.render_gallery(nullable)

    nonfinite = deepcopy(gallery)
    nonfinite["concepts"][0]["result"]["axial_thrust_n"] = math.inf
    with pytest.raises(ValueError, match="non-finite gallery value"):
        GENERATOR.validate_gallery(nonfinite)
    with pytest.raises(ValueError, match="objective value must be finite"):
        GENERATOR._finite_number(math.nan)


def test_selected_points_respect_conservation_and_fraction_bounds(gallery) -> None:
    for concept in gallery["concepts"]:
        point = concept["input"]
        result = concept["result"]
        fractions = point["charge_state_number_fractions"]
        assert abs(sum(fractions.values()) - 1.0) <= 2.0e-15
        assert point["mass_utilization_fraction_of_inlet_mass"] == pytest.approx(
            fractions["xe_plus"] + fractions["xe_double_plus"],
            rel=0.0,
            abs=2.0e-15,
        )
        assert 0.0 < point["beam_current_fraction_of_anode_current"] <= 1.0
        assert 0.0 <= point["axial_momentum_fraction_of_ion_momentum"] <= 1.0

        budget = result["power_budget"]
        assert all(
            value >= 0.0
            for key, value in budget.items()
            if key.endswith("_power_w") and key != "ppu_boundary_adjustment_w"
        )
        for key in (
            "anode_to_beam_efficiency",
            "thruster_electrical_to_beam_efficiency",
            "ppu_input_to_beam_efficiency",
        ):
            assert budget[key] is None or 0.0 <= budget[key] <= 1.0

        diagnostics = result["diagnostics"]
        assert abs(diagnostics["particle_rate_residual_particles_per_s"]) <= (
            4.0e-16 * max(1.0, result["total_xenon_particle_rate_per_s"])
        )
        assert abs(diagnostics["mass_flow_residual_kg_per_s"]) <= 3.0e-21
        assert abs(diagnostics["beam_current_residual_a"]) <= 2.0e-13
        assert abs(diagnostics["beam_power_residual_w"]) <= (
            5.0e-14 * max(1.0, budget["beam_kinetic_power_w"])
        )


def test_gallery_makes_no_unsupported_physical_claims(gallery) -> None:
    model = gallery["model"]
    assert model["dimensionality"] == "0D/global reduced performance"
    assert model["hypothetical_inputs"] is True
    assert "never validated physical thruster geometries" in model["interpretation"]
    assert any("No spatially resolved" in claim for claim in model["not_solved_at_l0"])
    assert any("No magnet radii" in claim for claim in model["not_solved_at_l0"])
    assert any(
        "No magnetic-field topology or plasma-discharge solution" in claim
        for claim in model["not_solved_at_l0"]
    )
    future = model["future_l1_candidate_fields"]
    assert future["status"].startswith("Required future evidence")
    assert len(future["fields"]) >= 6

    rendered = GENERATOR.render_gallery(gallery).lower()
    assert "validated design" not in rendered
    for concept in gallery["concepts"]:
        assert concept["entry_type"] == "representative operating point"
        assert (
            "operating concept" in concept["label"].lower()
            or "representative operating point" in concept["label"].lower()
        )
        concept_keys = json.dumps(concept, sort_keys=True).lower()
        assert "inner_magnet_radius_mm" not in concept_keys
        assert "outer_magnet_radius_mm" not in concept_keys
        assert "pareto" not in json.dumps(concept["selection"]).lower()
        assert "knee" not in json.dumps(concept["selection"]).lower()
        assert concept["caveats"]
