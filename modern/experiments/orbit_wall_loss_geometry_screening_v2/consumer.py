"""Consumers of v2: the coupling-v4.2 handoff (v1's consumer, imported) and the v1 dataset.

``verify_handoff`` / ``consume_handoff`` / ``consume_v4_export`` are the v1 screening's
first formal consumer of the orbit_mc coupling-v4.2 export format, reused by import with
attribution (``experiments.orbit_wall_loss_geometry_screening_v1.consumer``). v2 adds the
hash-bound loader of the v1 dataset for the pooled v1-vs-v2 comparison; the v1 rows are
never edited and never enter any v2 estimand.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from cft_revival.experiment_runtime.canonical import strict_json_loads

from experiments.orbit_wall_loss_geometry_screening_v1.consumer import (  # reused by import
    HANDOFF_CONSTANTS,
    HANDOFF_KEYS,
    HandoffConsumerError,
    consume_handoff,
    consume_v4_export,
    verify_handoff,
)

__all__ = [
    "HANDOFF_CONSTANTS",
    "HANDOFF_KEYS",
    "HandoffConsumerError",
    "consume_handoff",
    "consume_v4_export",
    "verify_handoff",
    "load_v1_dataset",
]


def load_v1_dataset(declaration: Mapping[str, Any], repository: Path) -> dict[str, Mapping[str, Any]]:
    """The v1 screening dataset rows by case id, bound by the v1 manifest and dataset bytes."""

    dataset_path = repository / declaration["dataset_path"]
    manifest_path = repository / declaration["experiment"] / "results" / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != declaration["results_manifest_file_sha256"]:
        raise HandoffConsumerError("v1 results manifest bytes differ from the declared authority")
    data = dataset_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != declaration["dataset_file_sha256"]:
        raise HandoffConsumerError("v1 dataset bytes differ from the declared authority")
    manifest = strict_json_loads(manifest_raw)
    if manifest.get("state") != "accepted_result":
        raise HandoffConsumerError("v1 bundle is not an accepted result")
    relative = "artifacts/geometry-wall-loss-dataset.json"
    entry = next((item for item in manifest["artifacts"] if item.get("path") == relative), None)
    if entry is None or entry["byte_sha256"] != declaration["dataset_file_sha256"]:
        raise HandoffConsumerError("v1 dataset is not the sealed manifest entry")
    dataset = strict_json_loads(data)
    if dataset.get("classification") != "SCREENING_L1A_FIELD_TEST_PARTICLE_WALL_LOSS" or dataset.get("evidentiary") is not True:
        raise HandoffConsumerError("v1 dataset is not the evidentiary screening dataset")
    rows = {row["case_id"]: row for row in dataset["designs"]}
    if len(rows) != dataset["design_count"]:
        raise HandoffConsumerError("v1 dataset rows are not unique")
    return rows
