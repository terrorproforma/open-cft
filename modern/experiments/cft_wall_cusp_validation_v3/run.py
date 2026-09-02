"""Launch, capture, and finalize the sole held-out worker attempt."""

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from .canonical import write_canonical, write_raw
from .control import (
    attempt_payload,
    host_runtime_payload,
    lock_acquisition_payload,
    phase_state_payload,
    protocol_snapshot_payload,
    query_gpu,
    run_control_preflight,
    wrapper_payload,
)
from .experiment import (
    ACCEPTED_COUPLING_COMMIT,
    PROTOCOL,
    PROTOCOL_SEMANTIC_SHA256,
    _runtime_identity,
    dependency_closure,
    finalize_attempt,
    run_serialization_preflight,
)


def main() -> None:
    experiment = Path(__file__).resolve().parent
    output = experiment / "results"
    if output.exists():
        raise RuntimeError("single execution output already exists")
    command = "python -m experiments.cft_wall_cusp_validation_v3.worker"
    closure = dependency_closure()
    control_preflight = run_control_preflight(
        protocol=PROTOCOL,
        protocol_semantic_sha256=PROTOCOL_SEMANTIC_SHA256,
        accepted_commit_sha=ACCEPTED_COUPLING_COMMIT,
    )
    domain_preflight = run_serialization_preflight()
    preflight = {
        "schema_version": "cft-wall-cusp-v3.combined-preflight/1.0.0",
        "clock": control_preflight["clock"],
        "status": "passed",
        "held_out_access_count": 0,
        "domain": domain_preflight,
        "control": control_preflight,
    }
    acquired = write_canonical(
        output / "execution-lock-acquired.canonical.json",
        lock_acquisition_payload(
            experiment_id=str(PROTOCOL["experiment_id"]),
            preregistration_commit_sha=str(closure["preregistration_commit_sha"]),
            protocol_semantic_sha256=PROTOCOL_SEMANTIC_SHA256,
            command=command,
            declared_device=str(PROTOCOL["maps"]["solver"]["device"]),
        ),
        exclusive=True,
    )
    write_raw(output / ".gitattributes", b"* -text\n", exclusive=True)
    write_raw(output / "access-log.jsonl", b"", exclusive=True)
    write_canonical(
        output / "protocol-snapshot.canonical.json",
        protocol_snapshot_payload(
            protocol=PROTOCOL,
            protocol_semantic_sha256=PROTOCOL_SEMANTIC_SHA256,
        ),
        exclusive=True,
    )
    write_canonical(
        output / "preflight-report.canonical.json",
        preflight,
        exclusive=True,
    )
    write_canonical(
        output / "dependency-closure.canonical.json",
        closure,
        exclusive=True,
    )
    write_canonical(
        output / "attempt-started.canonical.json",
        attempt_payload(acquired, state="started", exit_code=None),
        exclusive=True,
    )
    write_canonical(
        output / "phase-status.json",
        phase_state_payload(
            events=[
                {
                    "sequence": 1,
                    "clock": preflight["clock"],
                    "phase": "pre_access_initialization",
                    "status": "complete",
                    "payload": {
                        "held_out_access_count": 0,
                        "dependency_closure_semantic_sha256": closure[
                            "closure_semantic_sha256"
                        ],
                    },
                }
            ]
        ),
        exclusive=True,
    )
    environment = dict(os.environ)
    environment["CFT_ATTEMPT_COMMAND"] = command
    try:
        runtime = {
            **_runtime_identity(),
            "control_identity": host_runtime_payload(
                declared_device=str(PROTOCOL["maps"]["solver"]["device"]),
                gpu_query=query_gpu(),
            ),
        }
        runtime_stored = write_canonical(
            output / "runtime-identity.canonical.json",
            runtime,
            exclusive=True,
        )
        write_canonical(
            output / "initialization-wrapper.canonical.json",
            wrapper_payload(
                kind="pre_access_initialization",
                payload={
                    "lock_payload_sha256": acquired["semantic_integrity"][
                        "payload_sha256"
                    ],
                    "runtime_payload_sha256": runtime_stored[
                        "semantic_integrity"
                    ]["payload_sha256"],
                    "preflight_status": preflight["status"],
                    "held_out_access_count": 0,
                },
            ),
            exclusive=True,
        )
        with tempfile.TemporaryDirectory(
            prefix="cft-wall-cusp-v3-capture-"
        ) as temporary:
            stdout_path = Path(temporary) / "stdout.bin"
            stderr_path = Path(temporary) / "stderr.bin"
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                completed = subprocess.run(
                    (
                        sys.executable,
                        "-m",
                        "experiments.cft_wall_cusp_validation_v3.worker",
                    ),
                    cwd=experiment.parents[1],
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            stdout_bytes = stdout_path.read_bytes()
            stderr_bytes = stderr_path.read_bytes()
        return_code = completed.returncode
    except Exception:
        return_code = 1
        stdout_bytes = b""
        stderr_bytes = traceback.format_exc().encode("utf-8")
        if not (output / "runtime-identity.canonical.json").exists():
            write_canonical(
                output / "runtime-identity.canonical.json",
                {
                    "schema_version": "cft-wall-cusp-v3.runtime-failure/1.0.0",
                    "status": "runtime_initialization_failed",
                    "control_identity": host_runtime_payload(
                        declared_device=str(PROTOCOL["maps"]["solver"]["device"]),
                        gpu_query=query_gpu(),
                    ),
                },
                exclusive=True,
            )
    finalized = finalize_attempt(
        output,
        exit_code=return_code,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        command=command,
    )
    sys.stdout.buffer.write(stdout_bytes)
    sys.stderr.buffer.write(stderr_bytes)
    print(
        finalized.get("dataset", finalized.get("failure", {})).get("summary", {})
    )
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()

