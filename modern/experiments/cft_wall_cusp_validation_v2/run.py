"""Launch, capture, and finalize the sole held-out worker attempt."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .experiment import finalize_attempt


def main() -> None:
    experiment = Path(__file__).resolve().parent
    output = experiment / "results"
    if output.exists():
        raise RuntimeError("single execution output already exists")
    command = "python -m experiments.cft_wall_cusp_validation_v2.worker"
    environment = dict(os.environ)
    environment["CFT_ATTEMPT_COMMAND"] = command
    with tempfile.TemporaryDirectory(prefix="cft-wall-cusp-v2-capture-") as temporary:
        stdout_path = Path(temporary) / "stdout.log"
        stderr_path = Path(temporary) / "stderr.log"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                (sys.executable, "-m", "experiments.cft_wall_cusp_validation_v2.worker"),
                cwd=experiment.parents[1],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
    finalized = finalize_attempt(
        output,
        exit_code=completed.returncode,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        command=command,
    )
    sys.stdout.buffer.write(stdout_bytes)
    sys.stderr.buffer.write(stderr_bytes)
    print(
        finalized.get("dataset", finalized.get("failure", {})).get("summary", {})
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

