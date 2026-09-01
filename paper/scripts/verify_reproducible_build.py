"""Run two clean builds and require byte-identical PDFs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    build_script = repo / "paper/scripts/build.py"
    pdf = repo / "paper/build/manuscript.pdf"
    observations: list[dict[str, object]] = []
    for run in (1, 2):
        completed = subprocess.run(
            [sys.executable, str(build_script)],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)
            print(f"Clean build {run} failed.")
            return completed.returncode
        observations.append(
            {
                "run": run,
                "pdf_bytes": pdf.stat().st_size,
                "pdf_sha256": _sha256(pdf),
            }
        )

    identical = observations[0]["pdf_sha256"] == observations[1]["pdf_sha256"]
    report = {
        "document_type": "paper-two-clean-build-reproducibility-check",
        "schema_version": "1.0",
        "clean_builds": observations,
        "byte_identical": identical,
        "tool_versions": json.loads(
            (repo / "paper/build/tool-versions.json").read_text(encoding="utf-8")
        ),
    }
    output = repo / "paper/build/reproducibility-check.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not identical:
        print("Two clean builds produced different PDF SHA-256 values.")
        return 1
    print(
        "Two clean builds are byte-identical: "
        f"{observations[0]['pdf_sha256']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
