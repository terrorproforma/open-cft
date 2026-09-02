from pathlib import Path

import numpy as np

from cft_revival.experiment_runtime import BundleState
from experiments.l1a_field_surrogate_v5.protocol import (
    DEPENDENCY_LOCK,
    GEOMETRY_PREFLIGHT,
    PARTITIONS,
    SYNTHETIC_PREFLIGHT,
    verify_json,
)
from experiments.l1a_field_surrogate_v5.run import (
    ACCEPTED_RUNTIME_COMMIT,
    load_checkpoint,
    new_counters,
    save_checkpoint,
    static_undefined_names,
)


def test_sealed_preflight_is_complete_solver_free_and_runtime_owned() -> None:
    geometry = verify_json(GEOMETRY_PREFLIGHT)
    partitions = verify_json(PARTITIONS)
    synthetic = verify_json(SYNTHETIC_PREFLIGHT)
    dependency = verify_json(DEPENDENCY_LOCK)
    assert geometry["raw_count"] == 512
    assert geometry["frozen_hash_failure_count"] == 0
    assert len(partitions["frozen_raw_indices"]) == 240
    assert partitions["prior_coordinate_intersection_count"] == 0
    assert synthetic["passed"]
    assert synthetic["field_solver_access_count"] == 0
    assert synthetic["real_qoi_label_access_count"] == 0
    runtime = synthetic["runtime_path_coverage"]
    assert runtime["complete"]
    assert set(runtime["executed_paths"]) == set(runtime["required_groups"])
    assert all(runtime["executed_paths"].values())
    assert set(runtime["terminal_states_exercised"]) == {
        state.value for state in BundleState
    }
    assert runtime["shared_runtime_commit"] == ACCEPTED_RUNTIME_COMMIT
    assert dependency["accepted_experiment_runtime"]["commit"] == ACCEPTED_RUNTIME_COMMIT
    assert dependency["accepted_experiment_runtime"]["is_ancestor"]


def test_experiment_local_attributes_and_preflight_are_lf() -> None:
    root = GEOMETRY_PREFLIGHT.parent
    assert (root / ".gitattributes").read_text(encoding="utf-8").splitlines() == [
        ".gitattributes text eol=lf",
        "*.py text eol=lf",
        "*.md text eol=lf",
        "*.json text eol=lf",
        "*.sha256 text eol=lf",
        "*.npz binary",
    ]
    assert (Path(__file__).parent / ".gitattributes").read_text(
        encoding="utf-8"
    ).splitlines() == [".gitattributes text eol=lf", "*.py text eol=lf"]
    for path in (GEOMETRY_PREFLIGHT, PARTITIONS, SYNTHETIC_PREFLIGHT, DEPENDENCY_LOCK):
        assert b"\r\n" not in path.read_bytes()
        sidecar = path.with_name(path.name + ".sha256")
        assert b"\r\n" not in sidecar.read_bytes()
        verify_json(path)


def test_read_counter_increments_before_npz_open(tmp_path: Path) -> None:
    counters = new_counters()
    try:
        load_checkpoint(tmp_path, 0, "method", counters)
    except FileNotFoundError:
        pass
    assert counters["checkpoint_reads"]["method"] == 1
    assert counters["label_reads"]["method"] == 1
    vector = np.ones(8)
    qois = np.ones(9)
    save_checkpoint(tmp_path, 0, [0.0], vector, vector, qois, qois)
    loaded = load_checkpoint(tmp_path, 0, "method", counters)
    assert loaded["high_qois"].shape == (9,)
    assert counters["checkpoint_reads"]["method"] == 2


def test_static_checker_reports_unresolved_math(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def f(x):\n    return math.sqrt(x)\n", encoding="utf-8")
    result = static_undefined_names((bad,))
    assert not result["passed"]
    assert result["unresolved"] == {"bad.py": ["math"]}


def test_production_lifecycle_has_no_bespoke_lock_cache_or_failure_writer() -> None:
    source = (
        GEOMETRY_PREFLIGHT.parent / "run.py"
    ).read_text(encoding="utf-8")
    assert "ExperimentRuntime(" in source
    assert "RuntimeCallbacks(" in source
    for forbidden in (
        "def acquire_lock(",
        "shutil.rmtree",
        "cache.mkdir(",
        'write_json(RESULTS',
        "failure-manifest.json",
        "cleanup-record.json",
    ):
        assert forbidden not in source
