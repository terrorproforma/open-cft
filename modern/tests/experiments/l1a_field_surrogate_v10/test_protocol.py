import inspect
from pathlib import Path

import numpy as np
import pytest

from cft_revival.experiment_runtime import BundleState
from experiments.l1a_field_surrogate_v10.experiment import (
    SharedKernelGP,
    WeightedPOD,
    model_features,
    qoi_inverse,
    qoi_transform,
)
from experiments.l1a_field_surrogate_v10.protocol import (
    DEPENDENCY_LOCK,
    GEOMETRY_PREFLIGHT,
    PARTITIONS,
    PROTOCOL,
    SYNTHETIC_PREFLIGHT,
    verify_json,
)
from experiments.l1a_field_surrogate_v10.run import (
    ACCEPTED_RUNTIME_COMMIT,
    CACHE_RELATIVE,
    RESULT_RELATIVE,
    claim_git_common_attempt,
    execute,
    load_checkpoint,
    new_counters,
    save_checkpoint,
    static_undefined_names,
    validate_repository_status,
    verify_before_runtime,
)


def test_sealed_preflight_is_complete_solver_free_and_runtime_owned() -> None:
    geometry = verify_json(GEOMETRY_PREFLIGHT)
    partitions = verify_json(PARTITIONS)
    synthetic = verify_json(SYNTHETIC_PREFLIGHT)
    dependency = verify_json(DEPENDENCY_LOCK)
    assert geometry["raw_count"] == 1024
    assert geometry["frozen_hash_failure_count"] == 0
    assert len(partitions["frozen_raw_indices"]) == 432
    assert {name: len(rows) for name, rows in partitions["roles"].items()} == {
        "candidate": 270,
        "method": 54,
        "calibration": 54,
        "assessment": 54,
    }
    for role in ("method_strata", "calibration_strata", "assessment_strata"):
        assert {name: len(rows) for name, rows in partitions[role].items()} == {
            "interpolation": 18,
            "boundary": 18,
            "ood": 18,
        }
    assert all(
        set(counts.values()) == ({15} if role == "candidate" else {3})
        for role, counts in partitions["role_balance"].items()
    )
    assert {
        budget: set(counts.values())
        for budget, counts in partitions["candidate_prefix_balance"].items()
    } == {"162": {9}, "216": {12}, "270": {15}}
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


def test_audited_model_families_and_transforms_are_frozen() -> None:
    assert PROTOCOL["sampling"]["high_budgets"] == [162, 216, 270]
    assert PROTOCOL["models"]["scalar_families"] == [
        "reconstructed_physical_field_qois"
    ]
    assert PROTOCOL["models"]["field_families"] == [
        "localized_full_physical_residual_interpolator"
    ]
    coarse = {name: 1.2 for name in PROTOCOL["models"]["qois"]}
    ordinary = model_features([0.1] * 11, coarse, "field_energy_j")
    input_only = model_features([0.1] * 11, coarse, "source_representation_error")
    assert ordinary.shape == (12,)
    assert input_only.shape == (12,)
    for value in (1.0, 1.000001, 1.5):
        transformed = qoi_transform("minimum_mirror_ratio", value)
        assert qoi_inverse("minimum_mirror_ratio", transformed) == pytest.approx(value)


def test_ard_matern_fits_constant_mean_without_unscaled_fallback() -> None:
    x = np.asarray([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]])
    y = np.asarray([2.0, 2.1, 2.2])
    model = SharedKernelGP.fit(x, y, 1.0)
    assert model.ard_length.shape == (2,)
    assert model.y_mean.tolist() == pytest.approx([2.1])
    assert model.predict(x).shape == (3, 1)
    assert "exp(0)" not in inspect.getsource(
        __import__(
            "experiments.l1a_field_surrogate_v10.run",
            fromlist=["predict_scalar"],
        ).predict_scalar
    )


def test_primary_physical_representation_roundtrip_is_exact() -> None:
    rng = np.random.default_rng(9)
    snapshots = rng.normal(size=(2, 2 * 81 * 145))
    restored = WeightedPOD.representation_roundtrip(snapshots, 4)
    assert np.array_equal(restored, snapshots)


def test_local_interpolator_grid_and_heldout_policy_are_frozen() -> None:
    assert PROTOCOL["models"]["local_scopes"] == ["pooled", "stage-specific"]
    assert PROTOCOL["models"]["neighbour_counts"] == [8, 16]
    assert PROTOCOL["models"]["kernel_families"] == [
        "wendland-c2",
        "inverse-distance",
    ]
    module = __import__(
        "experiments.l1a_field_surrogate_v10.run",
        fromlist=["fit_development_candidates"],
    )
    source = inspect.getsource(module.fit_development_candidates)
    assert "fit_positions = np.flatnonzero(folds != fold)" in source
    assert "validation_positions = np.flatnonzero(folds == fold)" in source
    assert '"overlap_count"' in source


def test_production_lifecycle_uses_runtime_and_git_common_attempt_claim() -> None:
    source = (
        GEOMETRY_PREFLIGHT.parent / "run.py"
    ).read_text(encoding="utf-8")
    assert "ExperimentRuntime(" in source
    assert "RuntimeCallbacks(" in source
    execute_source = inspect.getsource(execute)
    assert execute_source.index("verified = verify_before_runtime()") < execute_source.index(
        "runtime = ExperimentRuntime("
    )
    assert execute_source.index("claim_git_common_attempt(verified)") < execute_source.index(
        "runtime = ExperimentRuntime("
    )
    claim_source = inspect.getsource(claim_git_common_attempt)
    assert "PinnedDirectory.open" in claim_source
    assert "open_file_exclusive" in claim_source
    assert "bind_preregistration" not in execute_source
    assert inspect.getsource(verify_before_runtime).count(
        '_git("status", "--porcelain=v1", "--untracked-files=all")'
    ) == 1
    for forbidden in (
        "def acquire_lock(",
        "shutil.rmtree",
        "cache.mkdir(",
        'write_json(RESULTS',
        "failure-manifest.json",
        "cleanup-record.json",
    ):
        assert forbidden not in source


def test_runtime_owned_result_and_cache_do_not_invalidate_attestation() -> None:
    validate_repository_status(
        "\n".join(
            (
                f"?? {RESULT_RELATIVE}/execution-lock.json",
                f"?? {CACHE_RELATIVE}/.experiment-cache.json",
            )
        ),
        allowed_untracked_roots=(RESULT_RELATIVE, CACHE_RELATIVE),
    )


@pytest.mark.parametrize(
    "status",
    (
        " M modern/experiments/l1a_field_surrogate_v10/predeclaration.json",
        "M  modern/experiments/l1a_field_surrogate_v10/run.py",
        "?? foreign-untracked.txt",
        "?? modern/experiments/l1a_field_surrogate_v10/results-sibling/file.json",
    ),
)
def test_protocol_or_foreign_repository_drift_invalidates_attestation(status: str) -> None:
    with pytest.raises(RuntimeError, match="repository drift"):
        validate_repository_status(
            status,
            allowed_untracked_roots=(RESULT_RELATIVE, CACHE_RELATIVE),
        )
