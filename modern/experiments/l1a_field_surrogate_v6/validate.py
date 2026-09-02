"""Shared-runtime validation entry point for field-surrogate v6."""

from __future__ import annotations

from typing import Any

from cft_revival.experiment_runtime import strict_json_file, validate_bundle

from .protocol import RESULTS


def validate_results() -> dict[str, Any]:
    manifest = validate_bundle(RESULTS)
    terminal = strict_json_file(RESULTS / "terminal.json")
    return {
        "passed": True,
        "state": manifest["state"],
        "counts": terminal["counts"],
        "payload": terminal["payload"],
        "primary_error": terminal["primary_error"],
        "secondary_errors": terminal["secondary_errors"],
        "artifact_count": manifest["artifact_count"],
    }
