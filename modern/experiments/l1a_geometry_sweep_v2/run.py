"""Single-use execution entry point for preregistered sweep v2."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import warp as wp

from cft_revival.fields import (
    field_artifact,
    max_field_difference,
    solve_problem_cpu,
    solve_problem_warp,
    validate_field_artifact_file,
    write_field_artifact,
)
from cft_revival.geometry import canonical_json, deserialize_geometry

from .experiment import (
    CLASSIFICATION,
    DOMAIN,
    PARITY_INDICES,
    PROTOCOL,
    SOLVER,
    build_case,
    case_record,
    evaluate_terminal_gates,
    nondominated,
    representative_roles,
    sample_designs,
)
from .protocol import (
    PROTOCOL_PATH,
    ROOT,
    stable_hash,
    verify_sidecar,
    write_bytes,
    write_sealed,
)

RESULTS = ROOT / "results"
PREREGISTERED_PATHS = (
    "modern/experiments/l1a_geometry_sweep_v2/README.md",
    "modern/experiments/l1a_geometry_sweep_v2/DEVLOG.md",
    "modern/experiments/l1a_geometry_sweep_v2/LEARNING_SCRATCHPAD.md",
    "modern/experiments/l1a_geometry_sweep_v2/__init__.py",
    "modern/experiments/l1a_geometry_sweep_v2/experiment.py",
    "modern/experiments/l1a_geometry_sweep_v2/protocol.json",
    "modern/experiments/l1a_geometry_sweep_v2/protocol.json.sha256",
    "modern/experiments/l1a_geometry_sweep_v2/protocol.py",
    "modern/experiments/l1a_geometry_sweep_v2/run.py",
    "modern/experiments/l1a_geometry_sweep_v2/validate.py",
    "modern/experiments/l1a_geometry_sweep_v2/results/README.md",
    "modern/tests/experiments/l1a_geometry_sweep_v2/test_preregistration.py",
    "modern/tests/experiments/l1a_geometry_sweep_v2/test_results.py",
)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repo_root() -> Path:
    return Path(_git(ROOT, "rev-parse", "--show-toplevel"))


def verify_preregistration_state() -> tuple[Path, str]:
    repo = _repo_root()
    revision = _git(repo, "rev-parse", "HEAD")
    tracked = set(_git(repo, "ls-files", *PREREGISTERED_PATHS).splitlines())
    if tracked != set(PREREGISTERED_PATHS):
        missing = sorted(set(PREREGISTERED_PATHS) - tracked)
        raise RuntimeError(f"preregistered paths are not committed: {missing}")
    subprocess.run(
        ("git", "diff", "--quiet", "HEAD", "--", *PREREGISTERED_PATHS),
        cwd=repo,
        check=True,
    )
    if _git(repo, "log", "-1", "--format=%s") != "preregister L1a geometry sweep v2":
        raise RuntimeError("HEAD is not the required preregistration commit")
    return repo, revision


def environment_record(preregistration_sha: str) -> dict[str, Any]:
    wp.init()
    device = wp.get_device(PROTOCOL["execution"]["device"])
    nvidia = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [item.strip() for item in nvidia.split(",")]
    return {
        "code_revision": preregistration_sha,
        "gpu": {
            "requested_device": PROTOCOL["execution"]["device"],
            "warp_name": device.name,
            "architecture": f"sm_{device.arch}",
            "uuid": device.uuid,
            "nvidia_smi_name": fields[0],
            "driver_version": fields[1],
            "memory_mib": int(fields[2]),
        },
        "warp": {
            "version": wp.__version__,
            "cuda_toolkit_version": wp.get_cuda_toolkit_version(),
            "cuda_driver_runtime_version": wp.get_cuda_driver_version(),
        },
        "scalar": PROTOCOL["execution"]["scalar"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "floating_replay_policy": PROTOCOL["replay_contract"]["cuda_policy"],
    }


def _write_geometry(path: Path, geometry: Any) -> dict[str, Any]:
    file_sha = write_bytes(path, canonical_json(geometry.to_dict()).encode("utf-8"))
    loaded = deserialize_geometry(path.read_text(encoding="utf-8"))
    if loaded.canonical_sha256 != geometry.canonical_sha256:
        raise RuntimeError("geometry artifact reload mismatch")
    return {
        "path": path.name,
        "file_sha256": file_sha,
        "payload_sha256": geometry.canonical_sha256,
    }


def _parity(case: Any, cuda_field: Any) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    cpu_field = solve_problem_cpu(case.problem, SOLVER)
    elapsed = perf_counter() - started
    return (
        {
            "case_id": case.case_id,
            "cpu_backend": cpu_field.diagnostics.backend,
            "cuda_backend": cuda_field.diagnostics.backend,
            "differences": max_field_difference(cpu_field, cuda_field),
        },
        elapsed,
    )


def _archive_representatives(
    role_records: Sequence[Mapping[str, str]],
    cache: Path,
) -> list[dict[str, Any]]:
    roles_by_case: dict[str, list[str]] = {}
    for item in role_records:
        roles_by_case.setdefault(item["case_id"], []).append(item["role"])
    artifacts: list[dict[str, Any]] = []
    destination = RESULTS / "representatives"
    destination.mkdir(parents=True, exist_ok=True)
    for case_id in sorted(roles_by_case):
        with (cache / f"{case_id}.pickle").open("rb") as stream:
            case, field = pickle.load(stream)
        geometry_path = destination / f"{case_id}.geometry.json"
        geometry_record = _write_geometry(geometry_path, case.geometry)
        geometry_record["path"] = f"representatives/{geometry_path.name}"
        full = field_artifact(
            case.problem,
            SOLVER,
            field,
            map_stride=1,
            wall_radius_m=case.geometry.chamber.outer_radius_m,
        )
        downsampled = field_artifact(
            case.problem,
            SOLVER,
            field,
            map_stride=4,
            wall_radius_m=case.geometry.chamber.outer_radius_m,
        )
        full_path = destination / f"{case_id}.field-full.json"
        down_path = destination / f"{case_id}.field-downsampled.json"
        full_sha = write_field_artifact(full_path, full)
        down_sha = write_field_artifact(down_path, downsampled)
        validate_field_artifact_file(
            full_path,
            expected_file_sha256=full_sha,
            expected_payload_sha256=full["integrity"]["payload_sha256"],
        )
        validate_field_artifact_file(
            down_path,
            expected_file_sha256=down_sha,
            expected_payload_sha256=downsampled["integrity"]["payload_sha256"],
        )
        artifacts.append(
            {
                "case_id": case_id,
                "roles": sorted(roles_by_case[case_id]),
                "geometry": geometry_record,
                "full_field": {
                    "path": f"representatives/{full_path.name}",
                    "file_sha256": full_sha,
                    "payload_sha256": full["integrity"]["payload_sha256"],
                    "stride": 1,
                },
                "downsampled_field": {
                    "path": f"representatives/{down_path.name}",
                    "file_sha256": down_sha,
                    "payload_sha256": downsampled["integrity"]["payload_sha256"],
                    "stride": 4,
                },
            }
        )
    return artifacts


def _qoi_ranges(cases: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    names = (
        "centreline_bz_min_t",
        "centreline_bz_max_t",
        "centreline_abs_bz_peak_t",
        "centreline_mid_abs_bz_t",
        "minimum_mirror_ratio",
        "maximum_mirror_ratio",
        "axis_cusp_count",
        "axis_null_count",
        "stage_gradient_rms_t_per_m",
        "stage_gradient_max_abs_t_per_m",
        "boundary_to_peak_ratio",
        "field_energy_j",
        "source_representation_error",
        "topology_confidence",
        "field_peak_t",
        "relative_residual_l2",
        "flux_reconstruction_identity_t_per_m",
    )
    successful = [case for case in cases if case["status"] == "success"]
    return {
        name: [
            min(float(case["qois"][name]) for case in successful),
            max(float(case["qois"][name]) for case in successful),
        ]
        for name in names
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Preregistered L1a geometry sweep v2",
        "",
        f"- Preregistration commit: `{summary['preregistration_commit_sha']}`",
        f"- Classification: `{CLASSIFICATION}`",
        f"- Terminal acceptance: `{summary['terminal_status']}`",
        f"- Evaluated: {summary['evaluated_count']}",
        f"- Failed: {summary['failed_count']}",
        f"- Nondominated: {summary['nondominated_count']}",
        f"- Representative roles: {len(summary['representative_roles'])}",
        f"- Unique representative artifacts: {summary['unique_representative_count']}",
        "",
        "## Seven terminal gates",
        "",
    ]
    for gate in summary["terminal_gates"]:
        lines.append(
            f"- `{gate['gate_id']}`: {'PASS' if gate['passed'] else 'FAIL'} "
            f"({gate['failure_count']} failures; observed `{gate['observed']}`)"
        )
    lines.extend(("", "## QoI ranges", ""))
    for name, values in sorted(summary["qoi_ranges"].items()):
        lines.append(f"- `{name}`: {values[0]:.12g} to {values[1]:.12g}")
    lines.extend(("", "## Representative roles", ""))
    for role in summary["representative_roles"]:
        lines.append(f"- `{role['role']}`: `{role['case_id']}`")
    lines.extend(
        (
            "",
            "## Replay and claim boundary",
            "",
            "Sampling, geometry, source and configuration identities are bitwise hash-bound.",
            "CUDA floating output is not claimed or required to be bitwise reproducible;",
            "future replay uses the preregistered scale-aware tolerances. Artifact hashes",
            "identify this single run. The six CPU comparisons are independent parity evidence.",
            "",
            "This is L1a field-only screening. It is not a material-aware permanent-magnet",
            "model, propulsion calculation, hardware-valid prediction or build qualification.",
            "",
        )
    )
    return "\n".join(lines)


def _manifest_file(path: Path, kind: str, payload_sha256: str | None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(RESULTS)).replace("\\", "/"),
        "kind": kind,
        "file_sha256": verify_sidecar(path),
        "payload_sha256": payload_sha256,
    }


def run_once() -> dict[str, Any]:
    repo, preregistration_sha = verify_preregistration_state()
    del repo
    lock_path = RESULTS / "execution-lock.json"
    if lock_path.exists():
        raise RuntimeError("sweep-v2 execution is already claimed; rerun forbidden")
    unexpected = [
        path
        for path in RESULTS.rglob("*")
        if path.is_file() and path.name != "README.md"
    ]
    if unexpected:
        raise RuntimeError(f"results directory is not empty: {unexpected}")
    environment = environment_record(preregistration_sha)
    protocol_file_sha = verify_sidecar(PROTOCOL_PATH)
    started_at = datetime.now(timezone.utc).isoformat()
    write_sealed(
        lock_path,
        {
            "schema_version": "cft-revival.l1a-sweep-v2.execution-lock/1.0.0",
            "state": "claimed-once",
            "started_at_utc": started_at,
            "preregistration_commit_sha": preregistration_sha,
            "protocol_file_sha256": protocol_file_sha,
            "protocol_payload_sha256": PROTOCOL["integrity"]["payload_sha256"],
            "case_count": PROTOCOL["execution"]["case_count"],
            "device": PROTOCOL["execution"]["device"],
        },
    )
    cache = RESULTS / ".working"
    cache.mkdir()
    designs = sample_designs()
    cases: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for index, design in enumerate(designs):
        stage = "CASE_PREPARATION_FAILURE"
        try:
            case = build_case(design, index)
            stage = "SOLVER_FAILURE"
            started = perf_counter()
            field = solve_problem_warp(
                case.problem,
                device=PROTOCOL["execution"]["device"],
                config=SOLVER,
            )
            elapsed = perf_counter() - started
            timings.append(
                {
                    "case_id": case.case_id,
                    "phase": "primary_cuda_solve",
                    "wall_time_seconds": elapsed,
                }
            )
            stage = "ARTIFACT_FAILURE"
            record = case_record(case, field)
            cases.append(record)
            with (cache / f"{case.case_id}.pickle").open("wb") as stream:
                pickle.dump((case, field), stream, protocol=5)
            if index in PARITY_INDICES:
                stage = "PARITY_FAILURE"
                parity_record, cpu_elapsed = _parity(case, field)
                parity.append(parity_record)
                timings.append(
                    {
                        "case_id": case.case_id,
                        "phase": "independent_cpu_parity",
                        "wall_time_seconds": cpu_elapsed,
                    }
                )
        except Exception as error:
            cases.append(
                {
                    "case_id": f"l1a-gs-v2-{index:03d}",
                    "status": "failure",
                    "failure": {
                        "code": stage,
                        "message": str(error),
                        "retryable": False,
                    },
                    "design_id": design.design_id,
                    "sampling_provenance": design.provenance,
                    "design_values": {
                        variable.name: value
                        for variable, value in zip(
                            design.variables, design.values, strict=True
                        )
                    },
                    "classification": CLASSIFICATION,
                }
            )
    front = nondominated(cases)
    role_records = representative_roles(front) if front else ()
    representative_artifacts = _archive_representatives(role_records, cache)
    shutil.rmtree(cache)
    terminal_gates = evaluate_terminal_gates(cases, parity)
    failed_count = sum(case["status"] == "failure" for case in cases)
    terminal_accepted = failed_count == 0 and all(
        gate["passed"] for gate in terminal_gates
    )
    raw_payload = {
        "schema_version": "cft-revival.l1a-sweep-v2.raw-results/1.0.0",
        "classification": CLASSIFICATION,
        "preregistration_commit_sha": preregistration_sha,
        "protocol_file_sha256": protocol_file_sha,
        "protocol_payload_sha256": PROTOCOL["integrity"]["payload_sha256"],
        "sampling_design_ids": [design.design_id for design in designs],
        "cases": cases,
        "parity": parity,
        "runtime_diagnostics": {
            "policy": PROTOCOL["execution"]["timing_policy"],
            "records": timings,
        },
    }
    raw, raw_file_sha = write_sealed(RESULTS / "raw-results.json", raw_payload)
    summary_payload = {
        "schema_version": "cft-revival.l1a-sweep-v2.summary/1.0.0",
        "classification": CLASSIFICATION,
        "screening_level": "L1a_field_only_design_space_screening",
        "preregistration_commit_sha": preregistration_sha,
        "protocol_payload_sha256": PROTOCOL["integrity"]["payload_sha256"],
        "environment": environment,
        "requested_count": len(designs),
        "evaluated_count": len(designs) - failed_count,
        "failed_count": failed_count,
        "nondominated_count": len(front),
        "nondominated_case_ids": [case["case_id"] for case in front],
        "representative_roles": list(role_records),
        "unique_representative_count": len(representative_artifacts),
        "terminal_gates": list(terminal_gates),
        "terminal_status": "ACCEPTED" if terminal_accepted else "FAILED",
        "qoi_ranges": _qoi_ranges(cases) if failed_count < len(cases) else {},
        "replay_contract": PROTOCOL["replay_contract"],
        "raw_results_payload_sha256": raw["integrity"]["payload_sha256"],
    }
    summary, summary_file_sha = write_sealed(
        RESULTS / "summary.json", summary_payload
    )
    report_sha = write_bytes(
        RESULTS / "REPORT.md", _report(summary).encode("utf-8")
    )
    deterministic_files = [
        {
            "path": "execution-lock.json",
            "kind": "execution_lock",
            "file_sha256": verify_sidecar(lock_path),
            "payload_sha256": json.loads(lock_path.read_text())["integrity"][
                "payload_sha256"
            ],
        },
        {
            "path": "raw-results.json",
            "kind": "raw_results",
            "file_sha256": raw_file_sha,
            "payload_sha256": raw["integrity"]["payload_sha256"],
        },
        {
            "path": "summary.json",
            "kind": "summary",
            "file_sha256": summary_file_sha,
            "payload_sha256": summary["integrity"]["payload_sha256"],
        },
        {
            "path": "REPORT.md",
            "kind": "report",
            "file_sha256": report_sha,
            "payload_sha256": None,
        },
    ]
    for artifact in representative_artifacts:
        for kind in ("geometry", "full_field", "downsampled_field"):
            item = artifact[kind]
            deterministic_files.append(
                {
                    "path": item["path"],
                    "kind": kind,
                    "file_sha256": item["file_sha256"],
                    "payload_sha256": item["payload_sha256"],
                }
            )
    manifest_payload = {
        "schema_version": "cft-revival.l1a-sweep-v2.manifest/1.0.0",
        "classification": CLASSIFICATION,
        "preregistration_commit_sha": preregistration_sha,
        "protocol_file_sha256": protocol_file_sha,
        "protocol_payload_sha256": PROTOCOL["integrity"]["payload_sha256"],
        "artifact_hash_policy": PROTOCOL["replay_contract"]["artifact_policy"],
        "raw_results_payload_sha256": raw["integrity"]["payload_sha256"],
        "summary_payload_sha256": summary["integrity"]["payload_sha256"],
        "representative_roles": list(role_records),
        "representative_artifacts": representative_artifacts,
        "deterministic_files": deterministic_files,
        "terminal_status": summary["terminal_status"],
    }
    manifest, _ = write_sealed(RESULTS / "manifest.json", manifest_payload)
    from .validate import validate_bundle

    validate_bundle(RESULTS)
    return {"raw": raw, "summary": summary, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run_once()
    summary = result["summary"]
    print(
        json.dumps(
            {
                "preregistration_commit_sha": summary["preregistration_commit_sha"],
                "terminal_status": summary["terminal_status"],
                "evaluated_count": summary["evaluated_count"],
                "failed_count": summary["failed_count"],
                "nondominated_count": summary["nondominated_count"],
                "unique_representative_count": summary[
                    "unique_representative_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if summary["terminal_status"] != "ACCEPTED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
