from __future__ import annotations

import copy
import hashlib
from math import nextafter

import pytest

import cft_revival.fields.numerics as numerics
from cft_revival.fields import (
    AxisymmetricDomain,
    AxisymmetricProblem,
    AzimuthalCurrentBand,
    FieldArtifactValidationError,
    FieldValidationError,
    SolverConfig,
    canonical_payload_sha256,
    design_manifest,
    field_artifact,
    manifest_entry,
    solve_problem_cpu,
    source_discretization_diagnostics,
    validate_design_manifest,
    validate_design_manifest_file,
    validate_field_artifact,
    validate_field_artifact_file,
    write_design_manifest,
    write_field_artifact,
)


def _domain() -> AxisymmetricDomain:
    return AxisymmetricDomain(0.12, -0.15, 0.15, 24, 48)


def _coil(name: str = "coil", ampere_turns: float = 2_000.0) -> AzimuthalCurrentBand:
    return AzimuthalCurrentBand(name, 0.04, 0.06, -0.02, 0.02, ampere_turns)


def _problem(name: str = "audit", *sources: AzimuthalCurrentBand) -> AxisymmetricProblem:
    return AxisymmetricProblem(name, _domain(), tuple(sources or (_coil(),)))


def _artifact(name: str = "audit", *sources: AzimuthalCurrentBand):
    problem = _problem(name, *sources)
    config = SolverConfig()
    return field_artifact(problem, config, solve_problem_cpu(problem, config))


def _reseal(value: dict[str, object]) -> None:
    payload = {key: item for key, item in value.items() if key != "integrity"}
    try:
        value["integrity"]["payload_sha256"] = canonical_payload_sha256(payload)
    except ValueError:
        # Nonfinite payloads are rejected before integrity validation.
        pass


@pytest.mark.parametrize(
    "arguments",
    [
        (1.0e-320, -1.0, 1.0, 4, 4),
        (1.0e308, -1.0e308, 1.0e308, 4, 4),
        (1.0, 1.0e308, nextafter(1.0e308, float("inf")), 4, 4),
    ],
)
def test_extreme_grid_spacing_failures_are_typed(arguments) -> None:
    with pytest.raises(FieldValidationError, match="derived"):
        AxisymmetricDomain(*arguments)


def test_bad_grid_count_type_is_typed_not_python_type_error() -> None:
    with pytest.raises(FieldValidationError, match="integer"):
        AxisymmetricDomain(1.0, -1.0, 1.0, "24", 48)


def test_boundary_adjacent_and_underresolved_sources_fail_before_sampling() -> None:
    with pytest.raises(FieldValidationError, match="interior dual-cell support"):
        AxisymmetricProblem(
            "boundary",
            _domain(),
            (AzimuthalCurrentBand("edge", 0.001, 0.03, -0.02, 0.02, 100),),
        )
    with pytest.raises(FieldValidationError, match="two grid spacings"):
        AxisymmetricProblem(
            "thin",
            _domain(),
            (AzimuthalCurrentBand("thin", 0.04, 0.045, -0.02, 0.02, 100),),
        )


def test_exact_two_spacing_source_passes_but_next_narrower_float_fails() -> None:
    domain = AxisymmetricDomain(0.1, 0.0, 0.1, 10, 10)
    exact = AxisymmetricProblem(
        "exact-two-cell",
        domain,
        (AzimuthalCurrentBand("exact", 0.04, 0.06, 0.04, 0.06, 100),),
    )
    represented = source_discretization_diagnostics(exact)[0]
    assert represented["radial_nodes_touched"] == 3
    assert represented["axial_nodes_touched"] == 3
    artifact = field_artifact(exact, SolverConfig(), solve_problem_cpu(exact))
    serialized_resolution = artifact["diagnostics"]["source_discretization"][0]
    assert serialized_resolution["radial_nodes_touched"] == 3
    assert serialized_resolution["axial_nodes_touched"] == 3
    validate_field_artifact(artifact)

    with pytest.raises(FieldValidationError, match="radial thickness"):
        AxisymmetricProblem(
            "radial-narrow",
            domain,
            (
                AzimuthalCurrentBand(
                    "narrow-r",
                    0.04,
                    nextafter(0.06, float("-inf")),
                    0.04,
                    0.06,
                    100,
                ),
            ),
        )
    with pytest.raises(FieldValidationError, match="axial thickness"):
        AxisymmetricProblem(
            "axial-narrow",
            domain,
            (
                AzimuthalCurrentBand(
                    "narrow-z",
                    0.04,
                    0.06,
                    0.04,
                    nextafter(0.06, float("-inf")),
                    100,
                ),
            ),
        )


def test_source_transfer_reports_geometry_error_and_conserves_current() -> None:
    problem = _problem("transfer", _coil())
    values = numerics.current_density_grid(problem)
    represented = sum(values) * problem.domain.dr_m * problem.domain.dz_m
    assert represented == pytest.approx(2_000.0, rel=2.0e-15)
    diagnostics = source_discretization_diagnostics(problem)
    assert len(diagnostics) == 1
    assert abs(diagnostics[0]["area_error_m2"]) < 1.0e-18
    assert diagnostics[0]["radial_nodes_touched"] >= 2
    assert diagnostics[0]["axial_nodes_touched"] >= 2
    assert abs(diagnostics[0]["centroid_r_error_m"]) <= 0.5 * problem.domain.dr_m
    assert abs(diagnostics[0]["centroid_z_error_m"]) <= 0.5 * problem.domain.dz_m
    assert diagnostics[0]["represented_signed_ampere_turns_a"] == pytest.approx(
        diagnostics[0]["requested_signed_ampere_turns_a"], rel=2.0e-15
    )


def test_zero_and_near_zero_fields_have_degenerate_topology_without_nulls() -> None:
    zero_problem = AxisymmetricProblem("zero", _domain())
    zero = field_artifact(
        zero_problem,
        SolverConfig(),
        solve_problem_cpu(zero_problem),
    )
    zero_topology = zero["summary"]["topology"]
    assert zero_topology["status"] == "degenerate_near_zero_field"
    assert zero_topology["axis_nulls"] == []
    assert zero_topology["axis_plateaus"] == []

    tiny = _artifact("tiny", _coil("tiny", 1.0e-12))
    tiny_topology = tiny["summary"]["topology"]
    assert tiny_topology["status"] == "degenerate_near_zero_field"
    assert tiny_topology["axis_nulls"] == []


def test_opposed_cusp_is_a_sign_changing_null_not_boundary_minimum() -> None:
    problem = AxisymmetricProblem(
        "cusp",
        _domain(),
        (
            AzimuthalCurrentBand("left", 0.04, 0.06, -0.07, -0.03, 2_000, 1),
            AzimuthalCurrentBand("right", 0.04, 0.06, 0.03, 0.07, 2_000, -1),
        ),
    )
    artifact = field_artifact(problem, SolverConfig(), solve_problem_cpu(problem))
    topology = artifact["summary"]["topology"]
    assert topology["status"] == "resolved_axis_nulls"
    assert topology["axis_nulls"][0]["kind"].startswith("sign_changing")
    assert "outer_boundary_b_magnitude_min_t" in artifact["summary"]


def test_false_recursive_crossing_restarts_from_true_residual(monkeypatch) -> None:
    original = numerics._recompute_true_residual
    calls = 0

    def first_recompute_is_above_tolerance(domain, rhs, solution):
        nonlocal calls
        calls += 1
        residual = original(domain, rhs, solution)
        if calls == 1:
            residual[domain.shape[1] + 1] += 1.0e-8
        return residual

    monkeypatch.setattr(
        numerics, "_recompute_true_residual", first_recompute_is_above_tolerance
    )
    field = solve_problem_cpu(_problem(), SolverConfig(max_true_residual_restarts=2))
    assert field.diagnostics.converged
    assert 1 <= field.diagnostics.true_residual_restarts <= 2
    assert not field.diagnostics.stagnation_detected


def test_true_residual_restart_exhaustion_is_explicit_stagnation(monkeypatch) -> None:
    original = numerics._recompute_true_residual

    def recompute_stays_above_tolerance(domain, rhs, solution):
        residual = original(domain, rhs, solution)
        residual[domain.shape[1] + 1] += 1.0e-6
        return residual

    monkeypatch.setattr(
        numerics, "_recompute_true_residual", recompute_stays_above_tolerance
    )
    field = solve_problem_cpu(
        _problem(),
        SolverConfig(max_true_residual_restarts=0),
        raise_on_nonconvergence=False,
    )
    assert not field.diagnostics.converged
    assert field.diagnostics.stagnation_detected
    assert field.diagnostics.true_residual_restarts == 0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.update({"unknown": 1}), "keys"),
        (lambda value: value["field_map"].update({"unknown": 1}), "keys"),
        (
            lambda value: value["field_map"].update({"downsample_stride": True}),
            "integer",
        ),
        (
            lambda value: value["field_map"]["r_m"].__setitem__(
                1, value["field_map"]["r_m"][0]
            ),
            "strictly increasing",
        ),
        (
            lambda value: value["field_map"]["z_m"].__setitem__(1, float("inf")),
            "finite",
        ),
        (lambda value: value["field_map"]["b_z_t"][0].pop(), "shape"),
        (
            lambda value: value["field_map"]["b_magnitude_t"][1].__setitem__(
                1, value["field_map"]["b_magnitude_t"][1][1] + 1.0e-4
            ),
            "inconsistent",
        ),
        (
            lambda value: value["diagnostics"].update({"converged": False}),
            "status",
        ),
        (
            lambda value: value["diagnostics"].update(
                {
                    "final_residual_l2": value["diagnostics"]["initial_residual_l2"],
                    "relative_residual_l2": 1.0,
                }
            ),
            "exceeds",
        ),
    ],
)
def test_strict_artifact_semantics_reject_corruption(mutator, message) -> None:
    artifact = _artifact()
    mutator(artifact)
    _reseal(artifact)
    with pytest.raises(ValueError, match=message):
        validate_field_artifact(artifact)


def test_payload_and_file_hashes_detect_tampering(tmp_path) -> None:
    artifact = _artifact()
    path = tmp_path / "artifact.json"
    digest = write_field_artifact(path, artifact)
    assert len(digest) == 64
    validate_field_artifact_file(
        path,
        expected_file_sha256=digest,
        expected_payload_sha256=artifact["integrity"]["payload_sha256"],
    )
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_field_artifact_file(path)


def test_huge_artifact_and_manifest_numbers_raise_typed_validation() -> None:
    with pytest.raises(FieldArtifactValidationError, match="canonical payload"):
        canonical_payload_sha256({"huge": 10**5000})

    artifact = _artifact()
    artifact["summary"]["b_magnitude_max_t"] = 10**400
    _reseal(artifact)
    with pytest.raises(FieldArtifactValidationError, match="finite binary64"):
        validate_field_artifact(artifact)

    entries = [
        {
            "name": "huge",
            "artifact": "huge.json",
            "artifact_file_sha256": "0" * 64,
            "artifact_payload_sha256": "1" * 64,
            "backend": "python",
            "iterations": 1,
            "relative_residual_l2": 0.0,
            "b_magnitude_min_t": 0.0,
            "b_magnitude_max_t": 10**400,
            "topology": {
                "status": "no_resolved_axis_null",
                "field_scale_t": 1.0,
                "null_tolerance_t": 1.0e-10,
                "axis_nulls": [],
                "axis_plateaus": [],
            },
        }
    ]
    with pytest.raises(FieldArtifactValidationError, match="finite binary64"):
        design_manifest(entries)


def test_huge_json_integer_loader_wraps_runtime_numeric_failure(tmp_path) -> None:
    path = tmp_path / "huge.json"
    data = ('{"value":' + "9" * 5000 + "}").encode("utf-8")
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    with pytest.raises(FieldArtifactValidationError, match="invalid JSON numeric"):
        validate_field_artifact_file(path)


def _write_two_artifacts(tmp_path):
    entries = []
    for name, polarity in (("first", 1), ("second", -1)):
        source = AzimuthalCurrentBand(
            f"{name}-coil", 0.04, 0.06, -0.02, 0.02, 2_000, polarity
        )
        artifact = _artifact(name, source)
        path = tmp_path / f"{name}.json"
        digest = write_field_artifact(path, artifact)
        entries.append(manifest_entry(path, artifact, digest))
    return entries


def test_manifest_anchors_artifacts_and_rejects_path_substitution(tmp_path) -> None:
    entries = _write_two_artifacts(tmp_path)
    manifest = design_manifest([entries[0]])
    path = tmp_path / "manifest.json"
    write_design_manifest(path, manifest)
    validate_design_manifest_file(path)

    substituted = copy.deepcopy(manifest)
    substituted["designs"][0]["artifact"] = entries[1]["artifact"]
    _reseal(substituted)
    write_design_manifest(path, substituted)
    with pytest.raises(ValueError, match="SHA-256"):
        validate_design_manifest_file(path)


def test_manifest_rejects_traversal_unknown_keys_and_payload_tampering(tmp_path) -> None:
    entries = _write_two_artifacts(tmp_path)
    traversal = design_manifest(entries)
    traversal["designs"][0]["artifact"] = "../first.json"
    _reseal(traversal)
    with pytest.raises(ValueError, match="plain filename"):
        validate_design_manifest(traversal)

    unknown = design_manifest(entries)
    unknown["designs"][0]["unknown"] = 1
    _reseal(unknown)
    with pytest.raises(ValueError, match="keys"):
        validate_design_manifest(unknown)

    tampered = design_manifest(entries)
    tampered["designs"][0]["name"] = "changed"
    with pytest.raises(ValueError, match="payload SHA-256"):
        validate_design_manifest(tampered)
