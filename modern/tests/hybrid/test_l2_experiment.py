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
