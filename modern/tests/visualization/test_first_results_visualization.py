"""Validation and smoke tests for the standalone first-results dashboard."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

MODERN = Path(__file__).resolve().parents[2]
GENERATOR_PATH = MODERN / "visualization" / "generate_first_results.py"
CHECKED_HTML = MODERN / "visualization" / "first-results.html"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "first_results_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


@pytest.fixture(scope="module")
def payload():
    return GENERATOR.build_payload()


def _assert_documented_rounding(actual: float, documented: float, decimals: int) -> None:
    tolerance = 0.5 * 10.0**-decimals
    assert abs(actual - documented) <= tolerance


def test_exact_count_columns_and_all_embedded_values_are_valid(payload) -> None:
    GENERATOR.validate_payload(payload)
    assert payload["sampleCount"] == 8192
    assert payload["sampling"] == {
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
    for values in payload["columns"].values():
        assert len(values) == 8192
        assert all(value is None or math.isfinite(value) for value in values)


def test_output_ranges_match_first_results_documented_rounding(payload) -> None:
    expected = {
        "thrust": (0.00188384225, 0.0513183291, 11, 10),
        "isp": (799.268670, 2726.81617, 6, 5),
        "beamCurrent": (0.100578251, 1.64624848, 9, 8),
        "anodePower": (21.5739213, 910.314451, 7, 6),
        "beamPower": (17.2688569, 786.015592, 7, 6),
        "ppuEfficiency": (0.379828910, 0.908991983, 9, 9),
    }
    for key, (minimum, maximum, min_decimals, max_decimals) in expected.items():
        result_range = payload["ranges"][key]
        _assert_documented_rounding(
            result_range["minimum"], minimum, min_decimals
        )
        _assert_documented_rounding(
            result_range["maximum"], maximum, max_decimals
        )


def test_default_and_reset_exact_bounds_retain_all_8192_points(payload) -> None:
    filter_keys = (
        "voltage",
        "massFlow",
        "utilization",
        "divergence",
        "doubleFraction",
        "thrust",
        "isp",
        "beamPower",
        "ppuEfficiency",
    )
    columns = payload["columns"]
    ranges = payload["ranges"]

    # Reproduce the former browser arithmetic: min + (max - min) can round
    # below the stored maximum and silently reject its owning point.
    reconstructed_upper = {
        key: ranges[key]["minimum"]
        + (ranges[key]["maximum"] - ranges[key]["minimum"])
        for key in filter_keys
    }
    former_count = sum(
        all(
            ranges[key]["minimum"] <= columns[key][index]
            <= reconstructed_upper[key]
            for key in filter_keys
        )
        for index in range(payload["sampleCount"])
    )
    assert former_count == 8191

    # Slider endpoints 0/1000 now use the stored extrema exactly. No epsilon
    # is added, so genuinely out-of-range values remain rejected.
    exact_count = sum(
        all(
            ranges[key]["minimum"] <= columns[key][index] <= ranges[key]["maximum"]
            for key in filter_keys
        )
        for index in range(payload["sampleCount"])
    )
    assert exact_count == payload["sampleCount"] == 8192
    assert not (
        ranges["thrust"]["minimum"]
        <= math.nextafter(ranges["thrust"]["maximum"], math.inf)
        <= ranges["thrust"]["maximum"]
    )


def test_selected_cases_match_first_results_documented_rounding(payload) -> None:
    columns = payload["columns"]
    expected = {
        0: (485.047280, 9.82608045e-7, 0.843437893, 0.0692342479,
            0.0186785330, 1938.39273, 319.667811),
        4096: (485.090005, 1.49106712e-6, 0.895308613, 0.0843043805,
               0.0310332192, 2122.31068, 520.707438),
        8191: (310.068642, 1.40578541e-6, 0.878519226, 0.0930025505,
               0.0226805104, 1645.17885, 311.206846),
    }
    decimal_places = (6, 14, 9, 10, 10, 5, 6)
    keys = (
        "voltage",
        "massFlow",
        "utilization",
        "doubleFraction",
        "thrust",
        "isp",
        "beamPower",
    )
    for index, values in expected.items():
        for key, documented, decimals in zip(
            keys, values, decimal_places, strict=True
        ):
            _assert_documented_rounding(
                columns[key][index], documented, decimals
            )


def test_generation_is_byte_deterministic(payload, tmp_path: Path) -> None:
    first = GENERATOR.render_html(payload)
    second = GENERATOR.render_html(GENERATOR.build_payload())
    assert first == second
    output = tmp_path / "first-results.html"
    GENERATOR.generate(output_path=output)
    assert output.read_text(encoding="utf-8") == first


def test_checked_html_is_current_self_contained_and_safe(payload) -> None:
    generated = GENERATOR.render_html(payload)
    assert CHECKED_HTML.read_text(encoding="utf-8") == generated
    assert '<script id="l0-data" type="application/json">' in generated
    assert "<canvas" in generated
    assert "8192" in generated
    assert "Pareto overlay is shown" in generated

    lowered = generated.lower()
    assert "fetch(" not in lowered
    assert "xmlhttprequest" not in lowered
    assert "websocket" not in lowered
    assert not re.search(r"""(?:src|href)\s*=\s*["'](?:[a-z]+:)?//""", generated, re.I)
    assert not re.search(r"\bhttps?://", generated, re.I)
    assert not re.search(r"\b[A-Za-z]:[\\/](?:Users|home)[\\/]", generated)
    assert not re.search(
        r"(?:api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*['\"][^'\"]+",
        generated,
        re.I,
    )


def test_generated_interaction_and_theme_listener_structure(payload) -> None:
    generated = GENERATOR.render_html(payload)

    assert (
        "function sliderValue(f,position){if(position===0)return f.lo;"
        "if(position===1000)return f.hi;"
    ) in generated
    assert "function rangeFor(f){return [sliderValue(" in generated

    assert '$("reset").addEventListener("click",()=>{' in generated
    assert "resetFilterValues();clearInteractionState();schedule()});" in generated
    assert (
        "function clearInteractionState(){cancelAnimationFrame(hoverFrame);"
        "hovered=-1;selected=-1;activeConcept=-1;pendingConcept=-1;"
        "showTooltip(-1);updateDetails();updateConceptCards();"
        "updateConceptStatus();drawScatter()}"
    ) in generated
    assert 'tip.style.display="none";tip.textContent=""' in generated

    assert 'const colorSchemeQuery=matchMedia("(prefers-color-scheme: dark)")' in generated
    assert "function handleColorSchemeChange(){drawAll()}" in generated
    assert 'colorSchemeQuery.addEventListener("change",handleColorSchemeChange)' in generated
    assert 'colorSchemeQuery.removeEventListener("change",handleColorSchemeChange)' in generated
    assert (
        'window.addEventListener("pagehide",removeColorSchemeListener,{once:true})'
        in generated
    )


def test_gallery_identities_and_indices_map_to_embedded_sweep(payload) -> None:
    gallery = payload["operatingConceptGallery"]
    source = gallery["source"]
    # Provenance pin: SHA-256 of the committed LF bytes of
    # config/l0-deterministic-sweep.json (was the CRLF-smudged hash
    # a4703ac1... before .gitattributes pinned eol=lf in fab0eccc).
    assert source["config_sha256"] == (
        "2d727b1af7d9be9f35f227cc318beae29af6cbd2fbead28842a4c17d67551b6b"
    )
    assert source["dataset_identity"]["sha256"] == (
        "c0a36ed83655d8bef0e8419a27dfbc330926716dadb6c893b6ef6f9b2ddbae84"
    )
    expected = {
        "maximum-axial-thrust": 6352,
        "maximum-specific-impulse": 2752,
        "minimum-anode-power-useful-thrust": 1633,
        "best-ppu-efficiency-useful-thrust": 148,
        "normalized-equal-weight-compromise": 1192,
    }
    assert {
        concept["concept_id"]: concept["index"]
        for concept in gallery["concepts"]
    } == expected
    for concept in gallery["concepts"]:
        index = concept["index"]
        assert concept["input"]["discharge_voltage_v"] == payload["columns"][
            "voltage"
        ][index]
        assert concept["input"]["propellant_mass_flow_kg_per_s"] == payload[
            "columns"
        ]["massFlow"][index]
        assert concept["result"]["axial_thrust_n"] == payload["columns"]["thrust"][
            index
        ]
        assert concept["result"]["specific_impulse_s"] == payload["columns"]["isp"][
            index
        ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("config", "config SHA-256"),
        ("dataset", "dataset SHA-256"),
        ("record", "input does not match index"),
    ),
)
def test_generation_rejects_stale_or_mismapped_gallery(
    payload, tmp_path: Path, mutation: str, message: str
) -> None:
    gallery = deepcopy(payload["operatingConceptGallery"])
    if mutation == "config":
        gallery["source"]["config_sha256"] = "0" * 64
    elif mutation == "dataset":
        gallery["source"]["dataset_identity"]["sha256"] = "f" * 64
    else:
        gallery["concepts"][0]["input"]["discharge_voltage_v"] += 1.0
    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(gallery), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        GENERATOR.build_payload(gallery_path=path)


def test_operating_concept_interaction_and_accessibility_structure(payload) -> None:
    generated = GENERATOR.render_html(payload)
    assert 'aria-labelledby="conceptTitle"' in generated
    assert 'id="conceptGrid" aria-label="Five representative L0 operating concepts"' in generated
    assert 'id="conceptStatus" role="status" aria-live="polite"' in generated
    assert 'id="showConcept" hidden>Reset filters and show point</button>' in generated
    assert 'button.setAttribute("aria-label",`Select ${conceptNames[' in generated
    assert 'button.setAttribute("aria-pressed","false")' in generated
    assert "navigateConceptCards(event,button)" in generated
    for key in ("ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp", "Home", "End"):
        assert f'event.key==="{key}"' in generated

    select_function = re.search(
        r"function selectConcept\(index\)\{(.*?)\}\nfunction resetFilterValues",
        generated,
    )
    assert select_function is not None
    selection_body = select_function.group(1)
    assert "selected=index" in selection_body
    assert "updateDetails()" in selection_body
    assert "drawScatter()" in selection_body
    assert "resetFilterValues" not in selection_body
    assert 'if(visible[index])focusScatter();else $("showConcept").focus()' in selection_body
    assert (
        '$("showConcept").addEventListener("click",()=>{'
        "const index=pendingConcept;if(index<0)return;resetFilterValues();"
    ) in generated
    assert "ctx.setLineDash(included?[]:[4,3])" in generated

    assert "Representative sampled operating points only—not 1D solutions" in generated
    assert "L1 axisymmetric field solutions are the next evidence-building step" in generated
    assert "no fabricated thruster drawings are shown" in generated
    assert "<svg" not in generated.lower()


def test_embedded_json_round_trips_without_nonstandard_constants(payload) -> None:
    generated = GENERATOR.render_html(payload)
    match = re.search(
        r'<script id="l0-data" type="application/json">(.*?)</script>',
        generated,
        re.DOTALL,
    )
    assert match is not None

    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant {value}")

    decoded = json.loads(match.group(1), parse_constant=reject_constant)
    assert decoded["sampleCount"] == 8192
    assert decoded["columns"]["thrust"] == payload["columns"]["thrust"]
