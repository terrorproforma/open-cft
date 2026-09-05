"""experiments/hybrid_l2_v2: protocol bindings to the PIC artifacts and the run -> finalize -> assess path on a synthetic field."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cft_revival.hybrid.cells import load_reference_partition
from cft_revival.pic2d.models import ChannelGeometry, Grid2D
from experiments.hybrid_l2_v2 import closure, run

MODERN = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def protocol() -> dict:
    return run.load_protocol()


def test_protocol_closures_and_reference_are_the_pic_artifacts(protocol: dict) -> None:
    grid = Grid2D(ChannelGeometry(0.002, 0.0, 0.024, 0.018, 0.003), 60, 480)
    partition = load_reference_partition(run.CATALOGUE_RESULTS, set_id="p2_divergent_exit", design_id="divergent-exit-stack", grid=grid,
                                         declared_cusp_planes_m=protocol["cells"]["declared_pic_cusp_planes_m"])
    reference = closure.build_pic_reference(partition, run.PIC_V2)
    assert np.allclose(reference["closures"]["cusp_conductance_s"], protocol["closures"]["cusp_conductance_s"], rtol=1e-12)
    assert np.allclose(reference["closures"]["leak_half_width_m"], protocol["closures"]["leak_half_width_m"], rtol=1e-12)
    for key, entry in protocol["pic_reference"]["quantities"].items():
        assert np.isclose(reference["quantities"][key]["reference"], entry["reference"], rtol=1e-12)
        assert reference["quantities"][key]["status"] == entry["status"]
        if entry["status"] == "compared":
            assert closure.TOLERANCE_FLOOR <= entry["tolerance"] <= closure.TOLERANCE_CAP
        else:
            assert entry["tolerance"] is None
    compared = [k for k, e in protocol["pic_reference"]["quantities"].items() if e["status"] == "compared"]
    assert {"discharge_current_a", "ionization_rate_per_s", "peak_n_e_per_m3", "exit_ion_beam_a", "step1_v"} <= set(compared)
    assert protocol["closures"]["provenance"]["base_maps_sha256"] == json.loads((run.PIC_V2 / "results" / "maps.npz.sha256.json").read_text())["byte_sha256"]


def test_cases_resolve_and_level_families_are_declared(protocol: dict) -> None:
    for name in protocol["cases"]:
        case = run.resolve_case(protocol, name)
        config = run.build_config(protocol, case)
        assert config.max_steps > 0
    for family in ("spatial_levels", "temporal_levels"):
        assert len(protocol["gates"][family]) == 3 and "base" in protocol["gates"][family]
        for name in protocol["gates"][family]:
            assert name in protocol["cases"]
    assert run.resolve_case(protocol, "closure-g-low")["conductance_scale"] == 0.7
    assert run.build_config(protocol, run.resolve_case(protocol, "closure-w-high")).leak_half_width_m[0] == pytest.approx(1.3 * protocol["closures"]["leak_half_width_m"][0])


def test_synthetic_run_finalize_assess_path(protocol: dict, tmp_path: Path) -> None:
    case = {**run.resolve_case(protocol, "spatial-coarse"), "name": "test-synthetic", "series_interval_steps": 5, "averaging_window_steps": 20,
            "checkpoint_every_steps": 20, "residual_window_steps": 20}
    results = tmp_path / "results-test"
    summary_path = run.run_case(protocol, case, results, field_kind="synthetic", max_steps=60, log=lambda _: None)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["stop_reason"] == "max_steps_reached" and summary["steps_completed"] == 60
    assert (results / "maps.npz").is_file() and (results / "series.npz").is_file() and (results / "l2-targets.json").is_file()
    assert (results / "checkpoint-final.json").is_file() and (results / "checkpoint-final.field.npz").is_file()
    assert summary["charge_identity_max_relative"] <= 1e-7
    quantities = run.l2_quantities(results)["quantities"]
    assert set(protocol["pic_reference"]["quantities"]) <= set(quantities)
    assessment = run.assess(protocol, cases=[("test-synthetic", results)], output=results / "assessment.json", log=lambda _: None,
                            require_reference_consistency=False)
    assert assessment["gate_l2"]["verdict"] == "not_evaluable"        # no plateau, no refinement families
    assert assessment["interface_conservation"]["checks"]["plateau"]["passed"] is False
    assert assessment["gate_l2"]["metrics"]["uncertainty_components"] == ["emulator", "input", "model_discrepancy", "numerical"]
    # a second run into the same directory is refused
    with pytest.raises(Exception, match="already holds a run"):
        run.run_case(protocol, case, results, field_kind="synthetic", max_steps=5, log=lambda _: None)
    shutil.rmtree(results)


def test_stop_file_and_resume_after_abrupt_stop(protocol: dict, tmp_path: Path) -> None:
    """The STOP file ends a run at the next series record with a checkpoint and no finalize; a run torn between
    checkpoints (series past the checkpoint step, torn last line) resumes from the checkpoint without duplicates."""
    case = {**run.resolve_case(protocol, "spatial-coarse"), "name": "test-resume", "series_interval_steps": 5, "averaging_window_steps": 20,
            "checkpoint_every_steps": 20, "residual_window_steps": 20}
    results = tmp_path / "results-resume"
    results.mkdir()
    (results / "STOP").write_text("", encoding="utf-8")
    logs: list[str] = []
    out = run.run_case(protocol, case, results, field_kind="synthetic", max_steps=60, log=logs.append)
    assert out == results / "checkpoint-latest.json" and not (results / "summary.json").exists() and not (results / "STOP").exists()
    assert json.loads(out.read_text(encoding="utf-8"))["step"] == 5 and any("STOP requested" in line for line in logs)
    series = results / "series.jsonl"
    assert [r["step"] for r in run._read_jsonl(series)] == [5]
    # emulate an abrupt stop of a later session: records past the checkpoint and a torn final line
    with series.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"step": 10, "fake": True}) + "\n" + json.dumps({"step": 15, "fake": True}) + "\n" + '{"step": 20, "torn')
    logs.clear()
    summary_path = run.run_case(protocol, case, results, field_kind="synthetic", max_steps=60, log=logs.append, resume=True)
    assert any("resumed results-resume at step 5" in line and "2 series records past the checkpoint dropped" in line for line in logs)
    records = run._read_jsonl(series)
    assert [r["step"] for r in records] == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] and not any(r.get("fake") for r in records)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["steps_completed"] == 60
    events = [r for r in run._read_jsonl(results / "status.jsonl") if r.get("event") in ("stop_requested", "resume")]
    assert [(e["event"], e["step"]) for e in events] == [("stop_requested", 5), ("resume", 5)] and events[-1]["series_records_dropped"] == 2
    assert [s["resumed_from_step"] for s in json.loads((results / "sessions.json").read_text(encoding="utf-8"))] == [0, 5]
    shutil.rmtree(results)
