"""Clean deterministic TeX build with no dependency installation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from check_paper import collect_errors


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _version(command: str) -> str:
    completed = subprocess.run(
        [command, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = (completed.stdout + completed.stderr).strip()
    return text.splitlines()[0] if text else "version output unavailable"


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _is_miktex(pdflatex: str) -> bool:
    return "miktex" in _version(pdflatex).casefold()


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    paper = repo / "paper"
    errors = collect_errors(repo)
    if errors:
        print("Build stopped because paper policy checks failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    config = json.loads((paper / "build-config.json").read_text(encoding="utf-8"))
    source_date_epoch = config.get("source_date_epoch")
    if isinstance(source_date_epoch, bool) or not isinstance(source_date_epoch, int):
        print("build-config.json has an invalid source_date_epoch.")
        return 1

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if pdflatex is None or bibtex is None:
        print(
            "A deterministic PDF build requires existing pdflatex and bibtex "
            "executables. Nothing was installed."
        )
        return 2

    build = paper / "build"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "SOURCE_DATE_EPOCH": str(source_date_epoch),
            "FORCE_SOURCE_DATE": "1",
            "TZ": "UTC",
            "MIKTEX_ENABLE_INSTALLER": "0",
        }
    )
    engine_flags = ["--disable-installer"] if _is_miktex(pdflatex) else []
    latex = [
        pdflatex,
        *engine_flags,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        "-output-directory=build",
        "manuscript.tex",
    ]
    _run(latex, paper, env)
    _run([bibtex, "build/manuscript"], paper, env)
    _run(latex, paper, env)
    _run(latex, paper, env)

    pdf = build / "manuscript.pdf"
    log = build / "manuscript.log"
    if not pdf.is_file() or pdf.stat().st_size == 0 or not log.is_file():
        print("TeX commands completed without a non-empty PDF and final log.")
        return 1
    log_text = log.read_text(encoding="utf-8", errors="replace")
    fatal_markers = (
        "LaTeX Error:",
        "There were undefined references",
        "There were undefined citations",
        "Overfull \\hbox",
        "Overfull \\vbox",
    )
    found = [marker for marker in fatal_markers if marker in log_text]
    if found:
        print("Final TeX log contains unresolved errors: " + ", ".join(found))
        return 1

    versions = {
        "document_type": "paper-build-tool-versions",
        "schema_version": "1.0",
        "source_date_epoch": source_date_epoch,
        "tools": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "pdflatex": _version(pdflatex),
            "bibtex": _version(bibtex),
        },
    }
    (build / "tool-versions.json").write_bytes(canonical_json(versions))
    pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    provenance = {
        "document_type": "paper-deterministic-build-provenance",
        "schema_version": "1.0",
        "evidence_revision": config["evidence_revision"],
        "source_date_epoch": source_date_epoch,
        "pdf": {
            "path": "paper/build/manuscript.pdf",
            "bytes": pdf.stat().st_size,
            "sha256": pdf_hash,
        },
        "tool_versions_path": "paper/build/tool-versions.json",
        "determinism_controls": config["determinism_controls"],
    }
    (build / "build-provenance.json").write_bytes(canonical_json(provenance))
    print(
        f"Built paper/build/manuscript.pdf ({pdf.stat().st_size} bytes, "
        f"SHA-256 {pdf_hash})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
