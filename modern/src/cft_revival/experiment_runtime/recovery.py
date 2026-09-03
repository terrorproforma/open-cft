"""Publish the completion manifest of an attempt whose artifacts and terminal record are durable.

Background. ``ExperimentRuntime._finalize_manifest`` builds the manifest inventory, pins every
file of the bundle (one read descriptor each) while the candidate is validated, and then
publishes ``manifest.json``. The pinning is bounded by the platform's descriptor limit (the
Windows CRT allows 8192 low-level handles); a bundle with more files than that could not be
published although every artifact, every hash sidecar, the transition log, the counters and
``terminal.json`` had already been written durably. The orbit wall-loss geometry screening v2
(16,957 files) exposed this on 2026-09-03.

This module offers the fail-closed recovery for exactly that situation: rebuild the manifest
from the durable, individually sidecar-attested files with the same inventory function the
runtime uses, hash the on-disk transition log, terminal record and lock, validate the candidate
with the runtime's own ``validate_bundle`` (unpinned) and publish it. Nothing is rerun and
nothing but ``manifest.json`` is written. The function refuses when a manifest already exists,
when the lock or the terminal record is missing, when the transition log does not end at
``terminal``, when the terminal state disagrees with the last transition, or when any file
disagrees with its sidecar (raised by the inventory). Its use must be disclosed by the
experiment that invokes it: the manifest is otherwise indistinguishable from one published by
the locked attempt, which is the point (same content), and the disclosure is what keeps the
record honest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .canonical import CANONICALIZATION_ID, canonical_bytes, strict_json_loads
from .contracts import MANIFEST_VERSION, validate_lock, validate_terminal
from .filesystem import (
    LOCK_NAME,
    MANIFEST_NAME,
    SIDECAR_SUFFIX,
    AtomicArtifactStore,
    FileOps,
    SafeRoot,
    canonical_bytes_from_parsed,
    pin_existing_root,
    platform_durability_contract,
    scan_tree,
)
from .lifecycle import LifecycleError, _inventory, _validate_transition, validate_bundle

RECOVERY_KIND = "manifest-published-after-descriptor-limit-failure"


def finalize_unpublished_attempt(
    root: SafeRoot | Path,
    *,
    approved_placeholders: Mapping[str, bytes] | None = None,
    ops: FileOps | None = None,
) -> dict[str, Any]:
    """Publish the manifest of a complete but unpublished attempt; return the manifest plus a recovery record."""

    owns_root = not isinstance(root, SafeRoot)
    safe = root if isinstance(root, SafeRoot) else pin_existing_root(root, ops)
    try:
        store = AtomicArtifactStore(safe, ops)
        safe.verify()
        entries = scan_tree(safe, ops)
        files = {name for name, kind in entries if kind == "file"}
        if MANIFEST_NAME in files:
            raise LifecycleError("bundle already has a completion manifest; nothing to recover")
        if LOCK_NAME not in files:
            raise LifecycleError("bundle has no execution lock; not a locked attempt")
        if "terminal.json" not in files or "terminal.json" + SIDECAR_SUFFIX not in files:
            raise LifecycleError("bundle has no durable terminal record; the attempt did not complete")
        lock_data = store.read_bytes(LOCK_NAME)
        lock = validate_lock(strict_json_loads(lock_data))
        terminal_data = store.read_bytes("terminal.json")
        terminal = validate_terminal(strict_json_loads(terminal_data))
        if canonical_bytes_from_parsed(terminal) != terminal_data:
            raise LifecycleError("terminal record is not canonical")
        event_paths = sorted(name for name in files if name.startswith("transitions/") and not name.endswith(SIDECAR_SUFFIX))
        events = [_validate_transition(strict_json_loads(store.read_bytes(item))) for item in event_paths]
        if not events or events[-1]["transition"] != "terminal":
            raise LifecycleError("transition log does not end at the terminal transition; the attempt did not complete")
        if events[-1]["details"].get("state") != terminal["state"]:
            raise LifecycleError("terminal transition and terminal record disagree")
        allowed_directories = {name for name, kind in entries if kind == "directory"}
        artifacts = _inventory(store, dict(approved_placeholders or {}), allowed_directories)
        identity = safe.identity
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "experiment_id": lock["experiment_id"],
            "state": terminal["state"],
            "manifest_is_sole_completion_marker": True,
            "canonicalization": CANONICALIZATION_ID,
            "lock_byte_sha256": hashlib.sha256(lock_data).hexdigest(),
            "transition_log_sha256": hashlib.sha256(canonical_bytes_from_parsed(events)).hexdigest(),
            "terminal_byte_sha256": hashlib.sha256(terminal_data).hexdigest(),
            "artifact_count": len(artifacts),
            "required_directories": sorted(allowed_directories),
            "root_identity": {
                "platform": identity.platform,
                "volume": identity.volume,
                "file_id": identity.file_id,
                "final_path_sha256": hashlib.sha256(identity.final_path.encode("utf-8")).hexdigest(),
            },
            "durability": {**platform_durability_contract(), "directory_flush_supported": safe.directory_flush_supported},
            "artifacts": artifacts,
        }
        candidate = canonical_bytes(manifest)
        validate_bundle(safe, approved_placeholders=approved_placeholders, manifest_override=candidate)
        store.write_manifest_bytes(candidate)
        published = store.read_bytes(MANIFEST_NAME)
        if published != candidate:
            raise LifecycleError("published manifest bytes differ from the validated candidate")
        return {
            "recovery": RECOVERY_KIND,
            "manifest_byte_sha256": hashlib.sha256(published).hexdigest(),
            "state": terminal["state"],
            "artifact_count": len(artifacts),
            "file_count": len(files),
            "terminal_byte_sha256": manifest["terminal_byte_sha256"],
            "transition_count": len(events),
            "manifest": manifest,
        }
    finally:
        if owns_root:
            safe.close()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args(argv)
    record = finalize_unpublished_attempt(arguments.root)
    record.pop("manifest")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
