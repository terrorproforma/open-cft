"""Generate the self-contained interactive dashboard for the first L0 sweep.

The data is reconstructed through the checked sweep and physics APIs.  The
generator deliberately excludes measured-at-generation runtime and timestamps
so identical source/configuration inputs produce byte-identical HTML.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cft_revival.physics.reference import evaluate_batch
from cft_revival.physics.workflows import (
    L0_MODEL_CLAIM,
    L0_MODEL_FIDELITY,
    operating_point_to_dict,
    result_to_dict,
    sweep_points_from_config,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "l0-deterministic-sweep.json"
DEFAULT_GALLERY = Path(__file__).resolve().with_name("design-gallery.json")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("first-results.html")

FIRST_RUN_PROVENANCE = {
    "date": "2026-09-01",
    "config": "config/l0-deterministic-sweep.json",
    "model_fidelity": L0_MODEL_FIDELITY,
    "artifact_schema_version": "1.0",
    "repository_commit": "not recorded in FIRST_RESULTS.md",
    "checked_backend": "NVIDIA Warp 1.14.0, CUDA Toolkit 12.9",
    "checked_device": "NVIDIA GeForce RTX 5090 (32,607 MiB, sm_120)",
    "os_python": "Windows 11 build 26200, Python 3.12.10",
    "driver": "NVIDIA 595.97; driver CUDA 13.2",
    "cuda_elapsed_seconds": 0.634302,
    "cuda_throughput_points_per_second": 12914.99,
    "python_reference_seconds": 0.141245,
    "timing_controlled": False,
}


def _read_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("sweep configuration must be a JSON object")
    return raw


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _dataset_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = sha256()
    digest.update(b"[")
    for position, record in enumerate(records):
        if position:
            digest.update(b",")
        digest.update(_canonical_json(record))
    digest.update(b"]")
    return digest.hexdigest()


def load_and_validate_gallery(
    gallery_path: Path,
    *,
    config_path: Path,
    config: Mapping[str, Any],
    sampling: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Load a gallery only when both source identities and records match."""

    gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
    if not isinstance(gallery, dict):
        raise ValueError("design gallery must be a JSON object")
    if gallery.get("document_type") != (
        "cft-revival-l0-representative-operating-point-gallery"
    ):
        raise ValueError("design gallery has an unexpected document_type")
    if gallery.get("schema_version") != "1.0":
        raise ValueError("design gallery has an unsupported schema_version")
    source = gallery.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("design gallery source must be an object")
    expected_config_sha256 = sha256(config_path.read_bytes()).hexdigest()
    if source.get("config_sha256") != expected_config_sha256:
        raise ValueError("design gallery config SHA-256 does not match the sweep config")
    if source.get("config") != config or source.get("sampling") != sampling:
        raise ValueError("design gallery source config or sampling metadata does not match")
    if source.get("sample_count") != len(records) or len(records) != 8192:
        raise ValueError("design gallery source sample count does not match")
    dataset_identity = source.get("dataset_identity")
    if not isinstance(dataset_identity, Mapping):
        raise ValueError("design gallery dataset identity must be an object")
    if dataset_identity.get("algorithm") != "sha256":
        raise ValueError("design gallery dataset identity must use SHA-256")
    expected_dataset_sha256 = _dataset_sha256(records)
    if dataset_identity.get("sha256") != expected_dataset_sha256:
        raise ValueError("design gallery dataset SHA-256 does not match the sweep")

    expected_ids = {
        "maximum-axial-thrust",
        "maximum-specific-impulse",
        "minimum-anode-power-useful-thrust",
        "best-ppu-efficiency-useful-thrust",
        "normalized-equal-weight-compromise",
    }
    concepts = gallery.get("concepts")
    if not isinstance(concepts, list) or len(concepts) != len(expected_ids):
        raise ValueError("design gallery must contain exactly five concepts")
    actual_ids: set[str] = set()
    indices: set[int] = set()
    for concept in concepts:
        if not isinstance(concept, Mapping):
            raise ValueError("design gallery concepts must be objects")
        concept_id = concept.get("concept_id")
        index = concept.get("index")
        if not isinstance(concept_id, str):
            raise ValueError("design gallery concept_id must be text")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
            raise ValueError("design gallery concept index is outside the sweep")
        if concept.get("input") != records[index]["input"]:
            raise ValueError(f"design gallery concept {concept_id!r} input does not match index")
        if concept.get("result") != records[index]["result"]:
            raise ValueError(f"design gallery concept {concept_id!r} result does not match index")
        actual_ids.add(concept_id)
        indices.add(index)
    if actual_ids != expected_ids or len(indices) != len(expected_ids):
        raise ValueError("design gallery concept identities or indices are invalid")
    return gallery


def _nested(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    return value


def _number(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"non-finite dashboard value: {value!r}")
    return converted


ColumnGetter = Callable[[Mapping[str, Any], Mapping[str, Any]], float | None]


def _input(path: str) -> ColumnGetter:
    return lambda point, _result: _number(_nested(point, path))


def _result(path: str) -> ColumnGetter:
    return lambda _point, result: _number(_nested(result, path))


COLUMN_SPECS: tuple[tuple[str, str, str, str, ColumnGetter], ...] = (
    ("voltage", "Discharge voltage", "V", "Input", _input("discharge_voltage_v")),
    (
        "massFlow",
        "Propellant mass flow",
        "kg/s",
        "Input",
        _input("propellant_mass_flow_kg_per_s"),
    ),
    (
        "neutralFraction",
        "Xe neutral number fraction",
        "1",
        "Input",
        _input("charge_state_number_fractions.xe_neutral"),
    ),
    (
        "plusFraction",
        "Xe+ number fraction",
        "1",
        "Input",
        _input("charge_state_number_fractions.xe_plus"),
    ),
    (
        "doubleFraction",
        "Xe2+ number fraction",
        "1",
        "Input",
        _input("charge_state_number_fractions.xe_double_plus"),
    ),
    (
        "utilization",
        "Mass utilization",
        "1",
        "Input",
        _input("mass_utilization_fraction_of_inlet_mass"),
    ),
    (
        "beamFraction",
        "Beam/anode current fraction",
        "1",
        "Input",
        _input("beam_current_fraction_of_anode_current"),
    ),
    (
        "divergence",
        "Axial/ion momentum fraction",
        "1",
        "Input",
        _input("axial_momentum_fraction_of_ion_momentum"),
    ),
    (
        "cathodePower",
        "Cathode input power",
        "W",
        "Input",
        _input("cathode_input_power_w"),
    ),
    (
        "requestedPpuPower",
        "Requested PPU input power",
        "W",
        "Input",
        _input("ppu_input_power_w"),
    ),
    ("xenonMass", "Xenon atom mass", "kg", "Input", _input("xenon_atom_mass_kg")),
    (
        "particleRate",
        "Total xenon particle rate",
        "particles/s",
        "Output",
        _result("total_xenon_particle_rate_per_s"),
    ),
    (
        "neutralRate",
        "Neutral particle rate",
        "particles/s",
        "Output",
        _result("neutral_particle_rate_per_s"),
    ),
    (
        "plusRate",
        "Xe+ particle rate",
        "particles/s",
        "Output",
        _result("xe_plus_particle_rate_per_s"),
    ),
    (
        "doubleRate",
        "Xe2+ particle rate",
        "particles/s",
        "Output",
        _result("xe_double_plus_particle_rate_per_s"),
    ),
    ("plusSpeed", "Xe+ speed", "m/s", "Output", _result("xe_plus_speed_m_per_s")),
    (
        "doubleSpeed",
        "Xe2+ speed",
        "m/s",
        "Output",
        _result("xe_double_plus_speed_m_per_s"),
    ),
    (
        "undivergedThrust",
        "Undiverged ion thrust",
        "N",
        "Output",
        _result("undiverged_ion_thrust_n"),
    ),
    ("thrust", "Axial thrust", "N", "Output", _result("axial_thrust_n")),
    ("isp", "Specific impulse", "s", "Output", _result("specific_impulse_s")),
    (
        "beamCurrent",
        "Beam current",
        "A",
        "Power",
        _result("power_budget.beam_current_a"),
    ),
    (
        "anodeCurrent",
        "Anode current",
        "A",
        "Power",
        _result("power_budget.anode_current_a"),
    ),
    (
        "beamPower",
        "Beam kinetic power",
        "W",
        "Power",
        _result("power_budget.beam_kinetic_power_w"),
    ),
    (
        "anodePower",
        "Anode input power",
        "W",
        "Power",
        _result("power_budget.anode_input_power_w"),
    ),
    (
        "reportedCathodePower",
        "Reported cathode input power",
        "W",
        "Power",
        _result("power_budget.cathode_input_power_w"),
    ),
    (
        "thrusterPower",
        "Thruster electrical input",
        "W",
        "Power",
        _result("power_budget.thruster_electrical_input_power_w"),
    ),
    (
        "reportedRequestedPpuPower",
        "Reported requested PPU input",
        "W",
        "Power",
        _result("power_budget.requested_ppu_input_power_w"),
    ),
    (
        "ppuPower",
        "Effective PPU input",
        "W",
        "Power",
        _result("power_budget.ppu_input_power_w"),
    ),
    (
        "ppuAdjustment",
        "PPU boundary adjustment",
        "W",
        "Power",
        _result("power_budget.ppu_boundary_adjustment_w"),
    ),
    (
        "ppuLoss",
        "PPU conversion loss",
        "W",
        "Power",
        _result("power_budget.ppu_conversion_loss_w"),
    ),
    (
        "anodeEfficiency",
        "Anode-to-beam efficiency",
        "1",
        "Power",
        _result("power_budget.anode_to_beam_efficiency"),
    ),
    (
        "thrusterEfficiency",
        "Thruster-electrical-to-beam efficiency",
        "1",
        "Power",
        _result("power_budget.thruster_electrical_to_beam_efficiency"),
    ),
    (
        "ppuEfficiency",
        "PPU-input-to-beam efficiency",
        "1",
        "Power",
        _result("power_budget.ppu_input_to_beam_efficiency"),
    ),
    (
        "particleResidual",
        "Particle-rate residual",
        "particles/s",
        "Diagnostic",
        _result("diagnostics.particle_rate_residual_particles_per_s"),
    ),
    (
        "massResidual",
        "Mass-flow residual",
        "kg/s",
        "Diagnostic",
        _result("diagnostics.mass_flow_residual_kg_per_s"),
    ),
    (
        "currentResidual",
        "Beam-current residual",
        "A",
        "Diagnostic",
        _result("diagnostics.beam_current_residual_a"),
    ),
    (
        "powerResidual",
        "Beam-power residual",
        "W",
        "Diagnostic",
        _result("diagnostics.beam_power_residual_w"),
    ),
    (
        "ppuMargin",
        "PPU power margin",
        "W",
        "Diagnostic",
        _result("diagnostics.ppu_power_margin_w"),
    ),
)


def _range(values: Sequence[float | None]) -> dict[str, float]:
    finite = [value for value in values if value is not None]
    if not finite:
        raise ValueError("dashboard column cannot be entirely null")
    return {"minimum": min(finite), "maximum": max(finite)}


def build_payload(
    config_path: Path = DEFAULT_CONFIG,
    gallery_path: Path = DEFAULT_GALLERY,
) -> dict[str, Any]:
    """Reproduce all 8,192 points through the production Python physics API."""

    config = _read_config(config_path)
    points, sampling = sweep_points_from_config(config)
    results = evaluate_batch(points)
    columns: dict[str, list[float | None]] = {
        key: [] for key, _label, _unit, _group, _getter in COLUMN_SPECS
    }
    warning_sets: set[tuple[tuple[str, str], ...]] = set()
    records: list[dict[str, Any]] = []

    for index, (point, result) in enumerate(zip(points, results, strict=True)):
        point_record = operating_point_to_dict(point)
        result_record = result_to_dict(result)
        records.append({"index": index, "input": point_record, "result": result_record})
        for key, _label, _unit, _group, getter in COLUMN_SPECS:
            columns[key].append(getter(point_record, result_record))
        warning_sets.add(
            tuple(
                (str(warning["code"]), str(warning["message"]))
                for warning in result_record["applicability_warnings"]
            )
        )

    # This fraction is an input coordinate in the sweep config, but the model
    # stores charge-state fractions after converting it to absolute fractions.
    columns["doubleShare"] = [
        double_fraction / utilization
        if double_fraction is not None and utilization not in (None, 0.0)
        else None
        for double_fraction, utilization in zip(
            columns["doubleFraction"], columns["utilization"], strict=True
        )
    ]
    columns["assumedPpuEfficiency"] = [
        thruster / requested
        if thruster is not None and requested not in (None, 0.0)
        else None
        for thruster, requested in zip(
            columns["thrusterPower"], columns["requestedPpuPower"], strict=True
        )
    ]

    metadata = [
        {"key": key, "label": label, "unit": unit, "group": group}
        for key, label, unit, group, _getter in COLUMN_SPECS
    ]
    metadata.extend(
        (
            {
                "key": "doubleShare",
                "label": "Xe2+ share of ions",
                "unit": "1",
                "group": "Input",
            },
            {
                "key": "assumedPpuEfficiency",
                "label": "Assumed PPU boundary efficiency",
                "unit": "1",
                "group": "Input",
            },
        )
    )
    ranges = {key: _range(values) for key, values in columns.items()}
    maximum_residuals = {
        key: max(abs(value) for value in columns[key] if value is not None)
        for key in ("particleResidual", "massResidual", "currentResidual", "powerResidual")
    }
    gallery = load_and_validate_gallery(
        gallery_path,
        config_path=config_path,
        config=config,
        sampling=sampling,
        records=records,
    )
    return {
        "documentType": "cft-revival-l0-first-results-visualization",
        "schemaVersion": "1.0",
        "modelFidelity": L0_MODEL_FIDELITY,
        "modelClaim": L0_MODEL_CLAIM,
        "hypotheticalInputs": True,
        "sampleCount": len(points),
        "sampling": sampling,
        "config": config,
        "provenance": FIRST_RUN_PROVENANCE,
        "columns": columns,
        "columnMetadata": metadata,
        "ranges": ranges,
        "maximumAbsoluteResiduals": maximum_residuals,
        "firstRunParity": {
            "comparedCount": 8192,
            "publishedNumericFields": 26,
            "mismatchCount": 0,
            "withinDocumentedBinary64Tolerance": True,
            "maximumAxialThrustDifferenceN": 6.93889e-18,
        },
        "applicabilityWarnings": [
            [{"code": code, "message": message} for code, message in warning_set]
            for warning_set in sorted(warning_sets)
        ],
        "pareto": {
            "included": False,
            "reason": (
                "No Pareto overlay is shown: these L0 outputs are not wired to "
                "the campaign optimization objectives, and the historical "
                "total_efficiency objective is not equivalent to any reported "
                "L0 efficiency boundary."
            ),
        },
        "operatingConceptGallery": gallery,
    }


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Reject malformed, non-finite, or count-inconsistent embedded data."""

    count = payload.get("sampleCount")
    if count != 8192:
        raise ValueError(f"expected exactly 8192 points, got {count!r}")
    columns = payload.get("columns")
    if not isinstance(columns, Mapping):
        raise ValueError("payload columns must be a mapping")
    for key, values in columns.items():
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"column {key!r} must contain exactly {count} values")
        for index, value in enumerate(values):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ValueError(f"invalid value at columns.{key}[{index}]")

    def walk(value: Any, path: str = "$") -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not isfinite(value):
                raise ValueError(f"non-finite value at {path}")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        raise ValueError(f"unsupported payload value at {path}: {type(value).__name__}")

    walk(payload)


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>First open-cft L0 results</title>
<style>
:root{color-scheme:light dark;--bg:#f5f7fb;--panel:#fff;--ink:#172033;--muted:#596579;--line:#d8deea;--accent:#175cd3;--warn:#8a4b08;--warnbg:#fff1d6;--plot:#f8faff;--focus:#8bb7ff}
@media(prefers-color-scheme:dark){:root{--bg:#0d1320;--panel:#151d2c;--ink:#ecf1fa;--muted:#a9b5c8;--line:#344056;--accent:#76a9ff;--warn:#ffd38b;--warnbg:#3a2b13;--plot:#101827;--focus:#8bb7ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1480px;margin:auto;padding:20px}
h1{font-size:clamp(1.55rem,3vw,2.5rem);margin:.15em 0}.eyebrow{text-transform:uppercase;letter-spacing:.1em;color:var(--accent);font-weight:750}.subtitle,.muted{color:var(--muted)}.warning{background:var(--warnbg);color:var(--warn);border:1px solid color-mix(in srgb,var(--warn) 35%,transparent);border-radius:10px;padding:12px 14px;font-weight:650;margin:14px 0}
.grid{display:grid;gap:14px}.headlines{grid-template-columns:repeat(4,minmax(0,1fr));margin:16px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 1px 3px #0001}.card{padding:14px}.card .value{font-size:clamp(1.08rem,2vw,1.45rem);font-weight:760;margin-top:5px;font-variant-numeric:tabular-nums}.card .label{color:var(--muted)}
.concept-section{margin:0 0 14px}.concept-heading{display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap}.concept-heading h2{margin:.1em 0}.fidelity-badge{border:1px solid var(--line);border-radius:999px;padding:3px 9px;color:var(--accent);font-weight:700}.concept-grid{display:grid;grid-template-columns:repeat(5,minmax(190px,1fr));gap:10px;margin-top:12px}.concept-card{display:block;width:100%;padding:12px;text-align:left;border:1px solid var(--line);border-radius:10px;background:var(--plot);color:var(--ink);cursor:pointer}.concept-card:hover{border-color:var(--accent);transform:translateY(-1px)}.concept-card[aria-pressed="true"]{border:2px solid var(--accent);padding:11px;box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 20%,transparent)}.concept-card h3{font-size:14px;margin:0 0 7px;color:var(--accent)}.concept-card p{font-size:12px;margin:5px 0}.concept-card .concept-rule{min-height:3.7em;color:var(--muted)}.concept-card .concept-metric{font-variant-numeric:tabular-nums}.concept-card .concept-caveat{border-top:1px solid var(--line);padding-top:7px;color:var(--warn)}.concept-status{display:flex;gap:10px;align-items:center;min-height:36px;margin-top:8px;color:var(--warn)}.concept-status:empty{min-height:0;margin:0}
.workspace{grid-template-columns:minmax(230px,280px) minmax(460px,1fr) minmax(270px,340px);align-items:start}.panel{padding:14px}.panel h2,.panel h3{margin:.1em 0 .65em}.filters{position:sticky;top:10px}.filter{padding:8px 0;border-top:1px solid var(--line)}.filter:first-of-type{border:0}.filter label{display:block;font-weight:650}.filter .readout{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}.slider-pair{height:28px;position:relative}.slider-pair input{position:absolute;left:0;top:2px;width:100%;accent-color:var(--accent);pointer-events:none}.slider-pair input::-webkit-slider-thumb{pointer-events:auto}.slider-pair input::-moz-range-thumb{pointer-events:auto}input,select,button{font:inherit}button,select{border:1px solid var(--line);color:var(--ink);background:var(--panel);border-radius:7px;padding:7px 9px}button{cursor:pointer}button:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible,input:focus-visible{outline:3px solid var(--focus);outline-offset:2px}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}.count{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums}
.canvas-wrap{height:clamp(360px,55vh,650px);position:relative;background:var(--plot);border:1px solid var(--line);border-radius:8px;overflow:hidden}.canvas-wrap canvas,.hist canvas{width:100%;height:100%;display:block}.tooltip{position:absolute;pointer-events:none;display:none;background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:7px;box-shadow:0 3px 12px #0003;font-size:12px;font-variant-numeric:tabular-nums;z-index:2}
.legend{height:10px;border-radius:10px;background:linear-gradient(90deg,#2a56c6,#20a4a1,#f6c844,#e34a33);margin-top:8px}.legend-labels{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}.hists{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:14px}.hist{height:145px}.hist-title{font-size:12px;color:var(--muted);font-weight:650;margin-bottom:4px}.details{max-height:790px;overflow:auto}.details table{width:100%;border-collapse:collapse;font-size:12px}.details th{padding:7px 4px;text-align:left;background:var(--panel);position:sticky;top:0;border-bottom:1px solid var(--line)}.details td{padding:5px 4px;border-bottom:1px solid var(--line);vertical-align:top}.details td:last-child{text-align:right;font-variant-numeric:tabular-nums}.group-row td{font-weight:750;color:var(--accent);padding-top:10px}
.provenance{grid-template-columns:1fr 1fr;margin-top:14px}.provenance dl{display:grid;grid-template-columns:minmax(120px,auto) 1fr;gap:6px 12px;margin:0}.provenance dt{color:var(--muted)}.provenance dd{margin:0;overflow-wrap:anywhere}.note{border-left:4px solid var(--accent);padding-left:10px}
@media(max-width:1200px){.concept-grid{grid-template-columns:repeat(3,minmax(210px,1fr))}}@media(max-width:1050px){.workspace{grid-template-columns:250px 1fr}.details{grid-column:1/-1;max-height:none}.headlines,.hists{grid-template-columns:repeat(2,1fr)}}@media(max-width:760px){.concept-grid{grid-template-columns:repeat(2,minmax(190px,1fr))}}@media(max-width:680px){main{padding:10px}.workspace,.provenance{grid-template-columns:1fr}.filters{position:static}.headlines,.hists,.concept-grid{grid-template-columns:1fr}.canvas-wrap{height:390px}.count{width:100%;margin:0}.concept-card .concept-rule{min-height:0}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style>
</head>
<body>
<main>
<div class="eyebrow">open-cft · first checked L0 sweep</div>
<h1>Conservation-reduced xenon performance space</h1>
<div class="subtitle">Explore 8,192 deterministic hypothetical operating points. Select or filter points to link every view.</div>
<div class="warning" role="alert">L0 / HYPOTHETICAL / NON-CALIBRATED — Numerical closure and CPU/CUDA parity establish implementation consistency only. These results are not measured-thruster validation or physically predictive calibration.</div>
<section class="grid headlines" aria-label="Headline ranges">
 <div class="card"><div class="label">Axial thrust</div><div class="value" id="thrustRange"></div></div>
 <div class="card"><div class="label">Specific impulse</div><div class="value" id="ispRange"></div></div>
 <div class="card"><div class="label">Beam kinetic power</div><div class="value" id="powerRange"></div></div>
 <div class="card"><div class="label">PPU-input-to-beam efficiency</div><div class="value" id="effRange"></div></div>
</section>
<section class="panel concept-section" aria-labelledby="conceptTitle" aria-describedby="conceptBoundary">
 <div class="concept-heading"><h2 id="conceptTitle">L0 operating concepts</h2><span class="fidelity-badge">0D / global · hypothetical</span></div>
 <p id="conceptBoundary">Representative sampled operating points only—not 1D solutions, physical magnet geometries, or validated thruster designs. L1 axisymmetric field solutions are the next evidence-building step; no fabricated thruster drawings are shown.</p>
 <div class="concept-grid" id="conceptGrid" aria-label="Five representative L0 operating concepts"></div>
 <div class="concept-status" id="conceptStatus" role="status" aria-live="polite"><span id="conceptStatusText"></span><button type="button" id="showConcept" hidden>Reset filters and show point</button></div>
</section>
<section class="grid workspace">
 <aside class="panel filters">
  <h2>Filters</h2>
  <div class="muted">Two handles set inclusive bounds.</div>
  <div id="filterList"></div>
  <button type="button" id="reset">Reset filters &amp; selection</button>
 </aside>
 <section>
  <div class="panel">
   <div class="controls">
    <label for="colorBy"><strong>Color by</strong></label>
    <select id="colorBy"><option value="beamPower">Beam kinetic power</option><option value="ppuEfficiency">PPU-input-to-beam efficiency</option></select>
    <span class="count" id="count" aria-live="polite"></span>
   </div>
   <div class="canvas-wrap" id="scatterWrap">
    <canvas id="scatter" tabindex="0" role="img" aria-label="Interactive scatter plot of axial thrust against specific impulse. Click or use pointer to select a point."></canvas>
    <div class="tooltip" id="tooltip" role="status"></div>
   </div>
   <div class="legend" aria-hidden="true"></div><div class="legend-labels"><span id="legendMin"></span><span id="legendTitle"></span><span id="legendMax"></span></div>
  </div>
  <div class="grid hists" aria-label="Filtered distributions">
   <div class="panel"><div class="hist-title">Axial thrust (N)</div><div class="hist"><canvas data-hist="thrust" role="img" aria-label="Axial thrust histogram"></canvas></div></div>
   <div class="panel"><div class="hist-title">Specific impulse (s)</div><div class="hist"><canvas data-hist="isp" role="img" aria-label="Specific impulse histogram"></canvas></div></div>
   <div class="panel"><div class="hist-title">Beam power (W)</div><div class="hist"><canvas data-hist="beamPower" role="img" aria-label="Beam power histogram"></canvas></div></div>
   <div class="panel"><div class="hist-title">PPU→beam efficiency</div><div class="hist"><canvas data-hist="ppuEfficiency" role="img" aria-label="Efficiency histogram"></canvas></div></div>
  </div>
 </section>
 <aside class="panel details" id="detailsPanel" tabindex="-1">
  <h2>Selected point</h2><div id="selectionHint" class="muted">Click a visible point to inspect every SI-explicit field.</div>
  <table aria-label="Selected operating point details"><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody id="detailsBody"></tbody></table>
 </aside>
</section>
<section class="grid provenance">
 <article class="panel"><h2>Provenance &amp; parity</h2><dl id="provenance"></dl></article>
 <article class="panel"><h2>Interpretation boundary</h2>
  <p class="note" id="paretoNote"></p>
  <p>L0 supplies charge-state mix, beam-current fraction, divergence, cathode power and PPU behavior as external inputs. It omits internal plasma, wall, thermal, topology, facility, erosion, uncertainty-calibration and experimental-comparison closures.</p>
  <p><strong>Timing is uncontrolled.</strong> The first CUDA observation includes preprocessing, allocations, transfers, synchronization and Python record construction. It is not a benchmark and supports no speedup claim.</p>
 </article>
</section>
</main>
<script id="l0-data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const raw=JSON.parse(document.getElementById("l0-data").textContent);
const C={}; for(const [k,v] of Object.entries(raw.columns)) C[k]=Float64Array.from(v,x=>x===null?NaN:x);
const N=raw.sampleCount, meta=new Map(raw.columnMetadata.map(x=>[x.key,x])), visible=new Uint8Array(N);
let selected=-1, hovered=-1, activeConcept=-1, pendingConcept=-1, filtered=[], frame=0, hoverFrame=0;
const filterDefs=[
 ["voltage","Voltage","V"],["massFlow","Mass flow","kg/s"],["utilization","Utilization","1"],
 ["divergence","Axial momentum fraction","1"],["doubleFraction","Xe2+ number fraction","1"],
 ["thrust","Axial thrust","N"],["isp","Specific impulse","s"],["beamPower","Beam power","W"],
 ["ppuEfficiency","PPU→beam efficiency","1"]
];
const filters=new Map(), $=id=>document.getElementById(id);
const finite=x=>Number.isFinite(x);
function fmt(x,unit=""){if(!finite(x))return "null";const a=Math.abs(x);let s=(a!==0&&(a<1e-3||a>=1e6))?x.toExponential(6):x.toLocaleString(undefined,{maximumSignificantDigits:8});return s+(unit&&unit!=="1"?" "+unit:"");}
function bounds(key){const r=raw.ranges[key];return [r.minimum,r.maximum]}
function sliderValue(f,position){if(position===0)return f.lo;if(position===1000)return f.hi;return f.lo+(f.hi-f.lo)*position/1000}
function addFilters(){const host=$("filterList");for(const [key,label,unit] of filterDefs){const [lo,hi]=bounds(key), step=(hi-lo)/1000||1;const box=document.createElement("div");box.className="filter";box.innerHTML=`<label>${label}</label><div class="readout"><span></span><span></span></div><div class="slider-pair"><input aria-label="${label} minimum" type="range" min="0" max="1000" value="0"><input aria-label="${label} maximum" type="range" min="0" max="1000" value="1000"></div>`;host.append(box);const inputs=box.querySelectorAll("input"),spans=box.querySelectorAll(".readout span");const filter={lo,hi,step,inputs,spans,unit};filters.set(key,filter);const sync=changed=>{let a=+inputs[0].value,b=+inputs[1].value;if(changed===0&&a>b){b=a;inputs[1].value=String(b)}if(changed===1&&b<a){a=b;inputs[0].value=String(a)}spans[0].textContent=fmt(sliderValue(filter,a),unit);spans[1].textContent=fmt(sliderValue(filter,b),unit);schedule()};inputs.forEach((input,i)=>input.addEventListener("input",()=>sync(i)));sync(0)}}
function rangeFor(f){return [sliderValue(f,+f.inputs[0].value),sliderValue(f,+f.inputs[1].value)]}
function applyFilters(){filtered=[];visible.fill(0);outer:for(let i=0;i<N;i++){for(const [key,f] of filters){const [lo,hi]=rangeFor(f),v=C[key][i];if(!finite(v)||v<lo||v>hi)continue outer}visible[i]=1;filtered.push(i)}if(hovered>=0&&!visible[hovered]){hovered=-1;showTooltip(-1)}$("count").textContent=`${filtered.length.toLocaleString()} / ${N.toLocaleString()} points`;updateConceptStatus();if(selected>=0)updateDetails();drawAll()}
function setupCanvas(canvas){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);const w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);return {ctx,w:rect.width,h:rect.height}}
function theme(){const s=getComputedStyle(document.documentElement);return {ink:s.getPropertyValue("--ink").trim(),muted:s.getPropertyValue("--muted").trim(),line:s.getPropertyValue("--line").trim(),accent:s.getPropertyValue("--accent").trim(),warn:s.getPropertyValue("--warn").trim(),plot:s.getPropertyValue("--plot").trim()}}
function turbo(t){t=Math.max(0,Math.min(1,t));const stops=[[42,86,198],[32,164,161],[246,200,68],[227,74,51]],p=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),f=p-i,a=stops[i],b=stops[i+1];return `rgb(${a.map((x,j)=>Math.round(x+(b[j]-x)*f)).join(",")})`}
const scatter=$("scatter"),margin={l:62,r:18,t:18,b:48};
function scatterGeometry(w,h){const [xmin,xmax]=bounds("isp"),[ymin,ymax]=bounds("thrust");return {x:i=>margin.l+(C.isp[i]-xmin)/(xmax-xmin)*(w-margin.l-margin.r),y:i=>h-margin.b-(C.thrust[i]-ymin)/(ymax-ymin)*(h-margin.t-margin.b),xmin,xmax,ymin,ymax}}
function drawAxes(ctx,w,h,g,t){ctx.strokeStyle=t.line;ctx.fillStyle=t.muted;ctx.lineWidth=1;ctx.font="11px system-ui";ctx.textAlign="center";ctx.beginPath();ctx.moveTo(margin.l,margin.t);ctx.lineTo(margin.l,h-margin.b);ctx.lineTo(w-margin.r,h-margin.b);ctx.stroke();for(let j=0;j<=4;j++){const x=margin.l+j*(w-margin.l-margin.r)/4,v=g.xmin+j*(g.xmax-g.xmin)/4;ctx.fillText(fmt(v),x,h-margin.b+17);const y=h-margin.b-j*(h-margin.t-margin.b)/4,v2=g.ymin+j*(g.ymax-g.ymin)/4;ctx.textAlign="right";ctx.fillText(fmt(v2),margin.l-7,y+4);ctx.textAlign="center"}ctx.fillText("Specific impulse (s)",(margin.l+w-margin.r)/2,h-9);ctx.save();ctx.translate(15,(margin.t+h-margin.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText("Axial thrust (N)",0,0);ctx.restore()}
function drawScatter(){const {ctx,w,h}=setupCanvas(scatter),t=theme(),g=scatterGeometry(w,h),color=$("colorBy").value,[cmin,cmax]=bounds(color);ctx.clearRect(0,0,w,h);ctx.fillStyle=t.plot;ctx.fillRect(0,0,w,h);drawAxes(ctx,w,h,g,t);ctx.globalAlpha=.62;for(const i of filtered){const z=(C[color][i]-cmin)/(cmax-cmin);ctx.fillStyle=turbo(z);ctx.fillRect(g.x(i)-1.5,g.y(i)-1.5,3,3)}ctx.globalAlpha=1;if(hovered>=0&&visible[hovered]){ctx.strokeStyle=t.muted;ctx.lineWidth=2;ctx.beginPath();ctx.arc(g.x(hovered),g.y(hovered),5,0,Math.PI*2);ctx.stroke()}if(selected>=0){const included=visible[selected]===1;ctx.setLineDash(included?[]:[4,3]);ctx.strokeStyle=included?t.ink:t.warn;ctx.fillStyle=included?t.accent:t.plot;ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(g.x(selected),g.y(selected),7,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.setLineDash([])}$("legendMin").textContent=fmt(cmin,meta.get(color).unit);$("legendMax").textContent=fmt(cmax,meta.get(color).unit);$("legendTitle").textContent=meta.get(color).label}
function drawHist(canvas,key){const {ctx,w,h}=setupCanvas(canvas),t=theme(),[lo,hi]=bounds(key),bins=new Uint32Array(30);for(const i of filtered){let b=Math.floor((C[key][i]-lo)/(hi-lo)*bins.length);b=Math.max(0,Math.min(bins.length-1,b));bins[b]++}const max=Math.max(1,...bins),pad=8,bw=(w-pad*2)/bins.length;ctx.clearRect(0,0,w,h);ctx.fillStyle=t.plot;ctx.fillRect(0,0,w,h);ctx.fillStyle=t.accent;for(let i=0;i<bins.length;i++){const bh=(h-28)*bins[i]/max;ctx.fillRect(pad+i*bw,h-18-bh,Math.max(1,bw-1),bh)}ctx.fillStyle=t.muted;ctx.font="10px system-ui";ctx.textAlign="left";ctx.fillText(fmt(lo),pad,h-4);ctx.textAlign="right";ctx.fillText(fmt(hi),w-pad,h-4)}
function drawAll(){drawScatter();document.querySelectorAll("[data-hist]").forEach(c=>drawHist(c,c.dataset.hist))}
function schedule(){cancelAnimationFrame(frame);frame=requestAnimationFrame(applyFilters)}
function nearest(clientX,clientY){const r=scatter.getBoundingClientRect(),g=scatterGeometry(r.width,r.height),x=clientX-r.left,y=clientY-r.top;let best=-1,d2=100;for(const i of filtered){const dx=g.x(i)-x,dy=g.y(i)-y,q=dx*dx+dy*dy;if(q<d2){d2=q;best=i}}return best}
function showTooltip(i,e){const tip=$("tooltip");if(i<0){tip.style.display="none";tip.textContent="";return}tip.innerHTML=`<strong>#${i}</strong><br>Thrust ${fmt(C.thrust[i],"N")}<br>Isp ${fmt(C.isp[i],"s")}`;tip.style.display="block";const wrap=$("scatterWrap").getBoundingClientRect();tip.style.left=Math.min(wrap.width-150,Math.max(5,e.clientX-wrap.left+10))+"px";tip.style.top=Math.min(wrap.height-70,Math.max(5,e.clientY-wrap.top+10))+"px"}
function clearHover(){cancelAnimationFrame(hoverFrame);hovered=-1;showTooltip(-1);drawScatter()}
function clearInteractionState(){cancelAnimationFrame(hoverFrame);hovered=-1;selected=-1;activeConcept=-1;pendingConcept=-1;showTooltip(-1);updateDetails();updateConceptCards();updateConceptStatus();drawScatter()}
scatter.addEventListener("pointermove",e=>{cancelAnimationFrame(hoverFrame);hoverFrame=requestAnimationFrame(()=>{hovered=nearest(e.clientX,e.clientY);showTooltip(hovered,e);drawScatter()})});scatter.addEventListener("pointerleave",clearHover);scatter.addEventListener("click",e=>{selected=nearest(e.clientX,e.clientY);activeConcept=-1;pendingConcept=-1;updateDetails();updateConceptCards();updateConceptStatus();drawScatter()});
function updateDetails(){const body=$("detailsBody");body.textContent="";if(selected<0){$("selectionHint").textContent="Click a visible point to inspect every SI-explicit field.";scatter.setAttribute("aria-label","Interactive scatter plot of axial thrust against specific impulse. Click or use pointer to select a point.");return}$("selectionHint").textContent=`Deterministic sweep index ${selected}${visible[selected]?"":" (excluded by current filters)"}`;scatter.setAttribute("aria-label",`Interactive thrust and specific-impulse scatter. Selected deterministic sweep point ${selected}${visible[selected]?"":"; excluded by current filters"}.`);let group="";for(const m of raw.columnMetadata){if(m.group!==group){group=m.group;const tr=document.createElement("tr");tr.className="group-row";tr.innerHTML=`<td colspan="2">${group}</td>`;body.append(tr)}const tr=document.createElement("tr"),v=C[m.key][selected];tr.innerHTML=`<td title="${m.key}">${m.label}</td><td>${fmt(v,m.unit)}</td>`;body.append(tr)}}
const conceptNames={"maximum-axial-thrust":"Maximum thrust","maximum-specific-impulse":"Maximum Isp","minimum-anode-power-useful-thrust":"Minimum anode power at/above median thrust","best-ppu-efficiency-useful-thrust":"Best PPU efficiency at/above median thrust","normalized-equal-weight-compromise":"Equal-weight normalized compromise"};
function conceptLine(className,label,text){const p=document.createElement("p");p.className=className;const strong=document.createElement("strong");strong.textContent=label+": ";p.append(strong,document.createTextNode(text));return p}
function initConcepts(){const host=$("conceptGrid");for(const concept of raw.operatingConceptGallery.concepts){const i=concept.index,button=document.createElement("button"),title=document.createElement("h3");button.type="button";button.className="concept-card";button.dataset.index=String(i);button.setAttribute("aria-pressed","false");button.setAttribute("aria-label",`Select ${conceptNames[concept.concept_id]}, deterministic sweep point ${i}`);title.textContent=conceptNames[concept.concept_id];button.append(title,conceptLine("concept-rule","Selection",concept.selection.rule),conceptLine("concept-metric","Inputs",`${fmt(C.voltage[i],"V")}; ${fmt(C.massFlow[i],"kg/s")}; beam fraction ${fmt(C.beamFraction[i])}; assumed PPU ${fmt(C.assumedPpuEfficiency[i])}`),conceptLine("concept-metric","Outputs",`thrust ${fmt(C.thrust[i],"N")}; Isp ${fmt(C.isp[i],"s")}; anode ${fmt(C.anodePower[i],"W")}; beam ${fmt(C.beamPower[i],"W")}; PPU→beam ${fmt(C.ppuEfficiency[i])}`),conceptLine("concept-metric","Charge & momentum",`Xe0 ${fmt(C.neutralFraction[i])}; Xe+ ${fmt(C.plusFraction[i])}; Xe2+ ${fmt(C.doubleFraction[i])}; utilization ${fmt(C.utilization[i])}; axial ${fmt(C.divergence[i])}`),conceptLine("concept-caveat","Caveat",concept.caveats[concept.caveats.length-1]));button.addEventListener("click",()=>selectConcept(i));button.addEventListener("keydown",event=>navigateConceptCards(event,button));host.append(button)}}
function navigateConceptCards(event,current){const cards=[...document.querySelectorAll(".concept-card")];let next=cards.indexOf(current);if(event.key==="ArrowRight"||event.key==="ArrowDown")next=(next+1)%cards.length;else if(event.key==="ArrowLeft"||event.key==="ArrowUp")next=(next-1+cards.length)%cards.length;else if(event.key==="Home")next=0;else if(event.key==="End")next=cards.length-1;else return;event.preventDefault();cards[next].focus()}
function updateConceptCards(){document.querySelectorAll(".concept-card").forEach(card=>card.setAttribute("aria-pressed",String(activeConcept===+card.dataset.index&&selected===activeConcept)))}
function updateConceptStatus(){const text=$("conceptStatusText"),action=$("showConcept");if(activeConcept<0||selected!==activeConcept){text.textContent="";action.hidden=true;return}if(!visible[activeConcept]){pendingConcept=activeConcept;text.textContent=`Concept point #${activeConcept} is excluded by current filters. Its dashed marker and details remain visible.`;action.hidden=false}else{pendingConcept=-1;text.textContent=`Selected concept point #${activeConcept}; current filters include it.`;action.hidden=true}}
function focusScatter(){const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;scatter.scrollIntoView({behavior:reduced?"auto":"smooth",block:"center"});scatter.focus({preventScroll:true})}
function selectConcept(index){cancelAnimationFrame(hoverFrame);hovered=-1;showTooltip(-1);activeConcept=index;selected=index;updateDetails();updateConceptCards();updateConceptStatus();drawScatter();if(visible[index])focusScatter();else $("showConcept").focus()}
function resetFilterValues(){for(const f of filters.values()){f.inputs[0].value="0";f.inputs[1].value="1000";f.inputs[0].dispatchEvent(new Event("input"))}}
function initSummary(){const set=(id,key,unit)=>{const [a,b]=bounds(key);$(id).textContent=`${fmt(a,unit)} – ${fmt(b,unit)}`};set("thrustRange","thrust","N");set("ispRange","isp","s");set("powerRange","beamPower","W");set("effRange","ppuEfficiency","");$("paretoNote").textContent=raw.pareto.reason;const p=raw.provenance,rows=[["Config",p.config],["Model",`${p.model_fidelity} · schema ${p.artifact_schema_version}`],["Commit",p.repository_commit],["Sample",`${raw.sampleCount.toLocaleString()} deterministic points · seed ${raw.sampling.seed}`],["First checked path",`${p.checked_backend}; ${p.checked_device}`],["Host",p.os_python],["Driver",p.driver],["Observed CUDA runtime",`${p.cuda_elapsed_seconds} s (${p.cuda_throughput_points_per_second.toLocaleString()} points/s; uncontrolled)`],["CPU reference",`${p.python_reference_seconds} s (separate, uncontrolled)`],["Parity",`${raw.firstRunParity.comparedCount.toLocaleString()} records × ${raw.firstRunParity.publishedNumericFields} fields; ${raw.firstRunParity.mismatchCount} mismatches`],["Max residuals",`particles ${fmt(raw.maximumAbsoluteResiduals.particleResidual,"particles/s")}; mass ${fmt(raw.maximumAbsoluteResiduals.massResidual,"kg/s")}; current ${fmt(raw.maximumAbsoluteResiduals.currentResidual,"A")}; power ${fmt(raw.maximumAbsoluteResiduals.powerResidual,"W")}`]];const dl=$("provenance");for(const [k,v] of rows){const dt=document.createElement("dt"),dd=document.createElement("dd");dt.textContent=k;dd.textContent=v;dl.append(dt,dd)}}
$("colorBy").addEventListener("change",drawScatter);$("reset").addEventListener("click",()=>{resetFilterValues();clearInteractionState();schedule()});
$("showConcept").addEventListener("click",()=>{const index=pendingConcept;if(index<0)return;resetFilterValues();activeConcept=index;selected=index;updateDetails();updateConceptCards();schedule();requestAnimationFrame(focusScatter)});
new ResizeObserver(()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(drawAll)}).observe(document.querySelector(".workspace"));
const colorSchemeQuery=matchMedia("(prefers-color-scheme: dark)");
function handleColorSchemeChange(){drawAll()}
function removeColorSchemeListener(){if(colorSchemeQuery.removeEventListener)colorSchemeQuery.removeEventListener("change",handleColorSchemeChange);else colorSchemeQuery.removeListener(handleColorSchemeChange)}
if(colorSchemeQuery.addEventListener)colorSchemeQuery.addEventListener("change",handleColorSchemeChange);else colorSchemeQuery.addListener(handleColorSchemeChange);
window.addEventListener("pagehide",removeColorSchemeListener,{once:true});
initSummary();initConcepts();addFilters();updateDetails();schedule();
</script>
</body>
</html>
"""
    return template.replace("__PAYLOAD__", encoded)


def generate(
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path = DEFAULT_OUTPUT,
    gallery_path: Path = DEFAULT_GALLERY,
) -> Path:
    payload = build_payload(config_path, gallery_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(payload), encoding="utf-8", newline="\n")
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = generate(
        args.config.resolve(),
        args.output.resolve(),
        args.gallery.resolve(),
    )
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
