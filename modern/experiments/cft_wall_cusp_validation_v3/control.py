"""Production control-envelope factories and failure-injection preflight."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .canonical import (
    CanonicalTypeError,
    CanonicalValueError,
    diagnose_bytes,
    load_canonical,
    semantic_hash,
    write_canonical,
    write_raw,
)


def clock_stamp() -> dict[str, Any]:
    return {
        "utc": datetime.now(timezone.utc),
        "monotonic_ns": time.monotonic_ns(),
    }


def host_runtime_payload(
    *,
    declared_device: str,
    gpu_query: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.runtime/1.0.0",
        "clock": clock_stamp(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "process_id": os.getpid(),
        "declared_device": declared_device,
        "gpu": gpu_query,
    }


def lock_acquisition_payload(
    *,
    experiment_id: str,
    preregistration_commit_sha: str,
    protocol_semantic_sha256: str,
    command: str,
    declared_device: str,
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.execution-lock-acquired/1.0.0",
        "experiment_id": experiment_id,
        "attempt": 1,
        "clock": clock_stamp(),
        "command": command,
        "host": platform.node(),
        "process_id": os.getpid(),
        "declared_device": declared_device,
        "preregistration_commit_sha": preregistration_commit_sha,
        "protocol_semantic_sha256": protocol_semantic_sha256,
        "clean_worktree_attested": True,
        "detached_head_attested": True,
        "status": "exclusive_lock_acquired",
    }


def lock_finalized_payload(
    acquired: Mapping[str, Any],
    *,
    exit_code: int,
    stdout_sha256: str,
    stderr_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.execution-lock-finalized/1.0.0",
        "acquired_lock_payload_sha256": acquired["semantic_integrity"][
            "payload_sha256"
        ],
        "attempt": acquired["attempt"],
        "clock": clock_stamp(),
        "exit_code": exit_code,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "status": "completed" if exit_code == 0 else "failed_immutable",
    }


def attempt_payload(
    acquired: Mapping[str, Any],
    *,
    state: str,
    exit_code: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.attempt/1.0.0",
        "attempt": 1,
        "clock": clock_stamp(),
        "state": state,
        "exit_code": exit_code,
        "acquired_lock_payload_sha256": acquired["semantic_integrity"][
            "payload_sha256"
        ],
    }


def dependency_closure_payload(
    *,
    preregistration_commit_sha: str,
    accepted_commit_sha: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.dependency-closure/1.0.0",
        "clock": clock_stamp(),
        "preregistration_commit_sha": preregistration_commit_sha,
        "accepted_coupling_commit_sha": accepted_commit_sha,
        "files": rows,
        "closure_semantic_sha256": semantic_hash(rows),
    }


def stream_metadata_payload(
    *,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.process-streams/1.0.0",
        "clock": clock_stamp(),
        "stdout": {
            "path": "stdout.bin",
            "byte_count": len(stdout),
            "byte_sha256": hashlib.sha256(stdout).hexdigest(),
        },
        "stderr": {
            "path": "stderr.bin",
            "byte_count": len(stderr),
            "byte_sha256": hashlib.sha256(stderr).hexdigest(),
        },
    }


def phase_state_payload(
    *,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.phase-state/1.0.0",
        "attempt": 1,
        "clock": clock_stamp(),
        "events": events,
    }


def failure_payload(
    *,
    phase: str,
    exception_type: str,
    message: str,
    traceback_sha256: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.failure/1.0.0",
        "clock": clock_stamp(),
        "status": "failed_immutable_no_patch_no_rerun",
        "failure": {
            "phase": phase,
            "exception_type": exception_type,
            "message": message,
            "traceback_sha256": traceback_sha256,
        },
        "summary": summary,
    }


def protocol_snapshot_payload(
    *,
    protocol: Mapping[str, Any],
    protocol_semantic_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.protocol-snapshot/1.0.0",
        "protocol_semantic_sha256": protocol_semantic_sha256,
        "protocol": protocol,
    }


def wrapper_payload(*, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.control-wrapper/1.0.0",
        "kind": kind,
        "payload": payload,
    }


def manifest_payload(
    *,
    experiment_id: str,
    preregistration_commit_sha: str,
    accepted_commit_sha: str,
    status: str,
    summary: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.manifest/1.0.0",
        "clock": clock_stamp(),
        "experiment_id": experiment_id,
        "attempt": 1,
        "preregistration_commit_sha": preregistration_commit_sha,
        "accepted_coupling_commit_sha": accepted_commit_sha,
        "status": status,
        "single_execution": True,
        "no_patch_or_rerun": True,
        "summary": summary,
        "artifacts": artifacts,
    }


def preflight_report_payload(
    *,
    writer_cases: list[dict[str, Any]],
    failure_injections: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "cft-wall-cusp-v3.production-writer-preflight/1.0.0",
        "clock": clock_stamp(),
        "status": "passed",
        "held_out_access_count": 0,
        "writer_cases": writer_cases,
        "failure_injections": failure_injections,
    }


def query_gpu() -> dict[str, Any]:
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version",
            "--format=csv,noheader",
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return {
        "command": "nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv,noheader",
        "exit_code": query.returncode,
        "rows": tuple(
            line.strip() for line in query.stdout.splitlines() if line.strip()
        ),
        "stderr_sha256": hashlib.sha256(query.stderr.encode("utf-8")).hexdigest(),
    }


def _write_case(
    root: Path,
    name: str,
    payload: Mapping[str, Any],
    *,
    exclusive: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / f"{name}.canonical.json"
    stored = write_canonical(path, payload, exclusive=exclusive)
    loaded = load_canonical(path)
    if loaded != stored:
        raise AssertionError(f"{name}: production write/load mismatch")
    return stored, {
        "name": name,
        "byte_count": len(path.read_bytes()),
        "byte_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "payload_sha256": stored["semantic_integrity"]["payload_sha256"],
    }


def run_control_preflight(
    *,
    protocol: Mapping[str, Any],
    protocol_semantic_sha256: str,
    accepted_commit_sha: str,
) -> dict[str, Any]:
    """Drive every pre-access control factory through the production writer."""

    with tempfile.TemporaryDirectory(prefix="cft-v3-writer-preflight-") as tmp:
        root = Path(tmp)
        rows: list[dict[str, Any]] = []
        acquired, row = _write_case(
            root,
            "lock-acquired",
            lock_acquisition_payload(
                experiment_id=str(protocol["experiment_id"]),
                preregistration_commit_sha="a" * 40,
                protocol_semantic_sha256=protocol_semantic_sha256,
                command="python -m experiments.cft_wall_cusp_validation_v3.worker",
                declared_device=str(protocol["maps"]["solver"]["device"]),
            ),
            exclusive=True,
        )
        rows.append(row)
        payloads = {
            "lock-finalized": lock_finalized_payload(
                acquired,
                exit_code=0,
                stdout_sha256="1" * 64,
                stderr_sha256="2" * 64,
            ),
            "attempt-started": attempt_payload(
                acquired,
                state="started",
                exit_code=None,
            ),
            "attempt-failed": attempt_payload(
                acquired,
                state="failed",
                exit_code=1,
            ),
            "runtime": host_runtime_payload(
                declared_device=str(protocol["maps"]["solver"]["device"]),
                gpu_query={"exit_code": 0, "rows": ("manufactured-gpu",)},
            ),
            "dependency-closure": dependency_closure_payload(
                preregistration_commit_sha="a" * 40,
                accepted_commit_sha=accepted_commit_sha,
                rows=[
                    {
                        "path": "manufactured.py",
                        "preregistration_git_blob_sha1": "b" * 40,
                        "accepted_baseline_git_blob_sha1": "b" * 40,
                    }
                ],
            ),
            "streams": stream_metadata_payload(
                stdout=b"manufactured stdout\r\n",
                stderr=b"manufactured stderr\n",
            ),
            "phase-empty": phase_state_payload(events=[]),
            "phase-started": phase_state_payload(
                events=[
                    {
                        "sequence": 1,
                        "clock": clock_stamp(),
                        "phase": "initialization",
                        "status": "complete",
                        "payload": {},
                    }
                ]
            ),
            "phase-complete": phase_state_payload(
                events=[
                    {
                        "sequence": 1,
                        "clock": clock_stamp(),
                        "phase": "manufactured",
                        "status": "complete",
                        "payload": {"outcome": "passed"},
                    }
                ]
            ),
            "phase-failed": phase_state_payload(
                events=[
                    {
                        "sequence": 1,
                        "clock": clock_stamp(),
                        "phase": "manufactured",
                        "status": "failed",
                        "payload": {"exception_type": "ManufacturedError"},
                    }
                ]
            ),
            "phase-skipped": phase_state_payload(
                events=[
                    {
                        "sequence": 1,
                        "clock": clock_stamp(),
                        "phase": "manufactured",
                        "status": "skipped",
                        "payload": {"reason": "dependency failed"},
                    }
                ]
            ),
            "failure": failure_payload(
                phase="manufactured",
                exception_type="ManufacturedError",
                message="injected",
                traceback_sha256="3" * 64,
                summary={"attempted_case_count": 0},
            ),
            "protocol-snapshot": protocol_snapshot_payload(
                protocol=protocol,
                protocol_semantic_sha256=protocol_semantic_sha256,
            ),
            "runtime-wrapper": wrapper_payload(
                kind="runtime",
                payload={"status": "manufactured"},
            ),
            "failure-wrapper": wrapper_payload(
                kind="failure",
                payload={"status": "manufactured"},
            ),
            "manifest": manifest_payload(
                experiment_id=str(protocol["experiment_id"]),
                preregistration_commit_sha="a" * 40,
                accepted_commit_sha=accepted_commit_sha,
                status="failed",
                summary={"attempted_case_count": 0},
                artifacts=[],
            ),
        }
        for name, payload in payloads.items():
            _, row = _write_case(root, name, payload)
            rows.append(row)
        write_raw(root / "stdout.bin", b"manufactured stdout\r\n", exclusive=True)
        write_raw(root / "stderr.bin", b"manufactured stderr\n", exclusive=True)
        write_raw(
            root / "traceback.bin",
            b"Traceback (most recent call last):\nManufacturedError\n",
            exclusive=True,
        )
        rows.extend(
            (
                {
                    "name": "stdout-raw",
                    "byte_count": (root / "stdout.bin").stat().st_size,
                    "byte_sha256": hashlib.sha256(
                        (root / "stdout.bin").read_bytes()
                    ).hexdigest(),
                },
                {
                    "name": "stderr-raw",
                    "byte_count": (root / "stderr.bin").stat().st_size,
                    "byte_sha256": hashlib.sha256(
                        (root / "stderr.bin").read_bytes()
                    ).hexdigest(),
                },
                {
                    "name": "traceback-raw",
                    "byte_count": (root / "traceback.bin").stat().st_size,
                    "byte_sha256": hashlib.sha256(
                        (root / "traceback.bin").read_bytes()
                    ).hexdigest(),
                },
            )
        )
        (root / "access-log.jsonl").open("xb").close()
        rows.append(
            {
                "name": "empty-access-log",
                "byte_count": 0,
                "byte_sha256": hashlib.sha256(b"").hexdigest(),
            }
        )

        injections: list[dict[str, Any]] = []
        acquired_original = (root / "lock-acquired.canonical.json").read_bytes()
        try:
            write_canonical(
                root / "lock-acquired.canonical.json",
                {"status": "second-attempt-must-fail"},
                exclusive=True,
            )
        except FileExistsError as error:
            injections.append(
                {
                    "name": "exclusive-lock-acquisition-conflict",
                    "passed": acquired_original
                    == (root / "lock-acquired.canonical.json").read_bytes(),
                    "diagnostic": type(error).__name__,
                }
            )

        unsupported_path = root / "unsupported.canonical.json"
        try:
            write_canonical(
                unsupported_path,
                {"outer": {"bad": Path("forbidden")}},
                exclusive=True,
            )
        except CanonicalTypeError as error:
            injections.append(
                {
                    "name": "canonicalize-before-exclusive-create",
                    "passed": not unsupported_path.exists()
                    and "$.outer.bad" in str(error),
                    "diagnostic": str(error),
                }
            )

        blocker = root / "blocker"
        blocker.write_bytes(b"not-a-directory")
        try:
            write_canonical(blocker / "child.json", {"ok": True})
        except OSError as error:
            injections.append(
                {
                    "name": "write-failure",
                    "passed": blocker.read_bytes() == b"not-a-directory",
                    "diagnostic": type(error).__name__,
                }
            )

        truncation = root / "truncated-lock.json"
        shutil.copyfile(root / "lock-acquired.canonical.json", truncation)
        original = truncation.read_bytes()
        truncation.write_bytes(original[: max(1, len(original) // 2)])
        diagnosis = diagnose_bytes(
            truncation.read_bytes(),
            source="injected-truncated-lock",
        )
        injections.append(
            {
                "name": "truncation-diagnosis",
                "passed": not diagnosis["canonical"]
                and diagnosis["byte_count"] < len(original),
                "diagnostic": diagnosis,
            }
        )

        acquired_before = (root / "lock-acquired.canonical.json").read_bytes()
        try:
            write_canonical(
                root / "bad-finalized-lock.json",
                {"clock": {"utc": datetime.now()}},
                exclusive=True,
            )
        except CanonicalValueError as error:
            injections.append(
                {
                    "name": "finalization-failure-preserves-acquired-lock",
                    "passed": acquired_before
                    == (root / "lock-acquired.canonical.json").read_bytes()
                    and not (root / "bad-finalized-lock.json").exists(),
                    "diagnostic": str(error),
                }
            )

        if not all(item["passed"] for item in injections):
            raise AssertionError("production writer failure injection failed")
        report = preflight_report_payload(
            writer_cases=rows,
            failure_injections=injections,
        )
        _, report_row = _write_case(root, "preflight-report", report)
        report["self_write"] = report_row
        return report
