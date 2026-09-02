"""Reusable one-attempt experiment lifecycle with fail-closed terminal bundles."""

from __future__ import annotations

import hashlib
import os
import platform
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from .canonical import CANONICALIZATION_ID, canonical_bytes, producer_id, strict_json_loads
from .contracts import (
    ACCESS_VERSION,
    COUNTER_VERSION,
    LOCK_VERSION,
    MANIFEST_VERSION,
    TERMINAL_VERSION,
    TRANSITION_VERSION,
    ContractError,
    bounded_int,
    exact_bool,
    exact_mapping,
    nonempty_string,
    sha256_string,
    validate_counts,
    validate_encoded_value,
    validate_lock,
    validate_terminal,
    validate_utc_tag,
)
from .filesystem import (
    LOCK_NAME,
    MANIFEST_NAME,
    SIDECAR_SUFFIX,
    TEMP_SUFFIX,
    AtomicArtifactStore,
    FileOps,
    FilesystemSafetyError,
    ManagedCache,
    RootPolicy,
    SafeRoot,
    _exclusive_durable_write,
    canonical_bytes_from_parsed,
    pin_existing_root,
    platform_durability_contract,
    preflight_result_root,
    relative_path,
    scan_tree,
    verify_pair,
)
from .platformfs import identities_overlap


class BundleState(str, Enum):
    PREBUNDLE_FAILURE = "prebundle_failure"
    RUNTIME_FAILURE = "runtime_failure"
    DEVELOPMENT_REJECTION = "development_rejection"
    ASSESSMENT_REJECTION = "assessment_rejection"
    ACCEPTED_RESULT = "accepted_result"


TERMINAL_STATES = frozenset(BundleState)
EVENT_TRANSITION_PAIRS = frozenset(
    {
        ("lock-acquired", "cache-prepared"),
        ("lock-acquired", "prebundle-failed"),
        ("cache-prepared", "prebundle-started"),
        ("cache-prepared", "prebundle-failed"),
        ("prebundle-started", "prebundle-completed"),
        ("prebundle-started", "prebundle-failed"),
        ("prebundle-completed", "development-started"),
        ("development-started", "development-accepted"),
        ("development-started", "development-rejected"),
        ("development-started", "runtime-failed"),
        ("development-accepted", "assessment-started"),
        ("assessment-started", "assessment-accepted"),
        ("assessment-started", "assessment-rejected"),
        ("assessment-started", "runtime-failed"),
        ("assessment-accepted", "runtime-failed"),
        ("assessment-rejected", "runtime-failed"),
        ("development-rejected", "runtime-failed"),
        ("runtime-failed", "cleanup-failed"),
        ("prebundle-failed", "cleanup-failed"),
        ("prebundle-failed", "terminal"),
        ("runtime-failed", "terminal"),
        ("cleanup-failed", "terminal"),
        ("development-rejected", "terminal"),
        ("assessment-rejected", "terminal"),
        ("assessment-accepted", "terminal"),
    }
)


class LifecycleError(RuntimeError):
    """Lifecycle execution or validation failed closed."""


class ExistingLockError(LifecycleError):
    """An immutable same, different, or malformed lock already exists."""

    def __init__(self, classification: str) -> None:
        super().__init__(f"execution lock already exists ({classification})")
        self.classification = classification


@dataclass(frozen=True)
class ExecutionAttestation:
    attempt: int
    commit: str
    command: str
    device: str
    clean_worktree: bool
    host: str = field(default_factory=platform.node)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.attempt) is not int or self.attempt < 1 or self.attempt > 2**63 - 1:
            raise LifecycleError("attempt must be an exact positive integer")
        if type(self.commit) is not str or not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", self.commit
        ):
            raise LifecycleError("commit must be a lowercase 40- or 64-hex identity")
        if any(
            type(value) is not str or not value.strip() or len(value) > 4096
            for value in (self.command, self.host, self.device)
        ):
            raise LifecycleError("command, host, and device must be bounded strings")
        if type(self.clean_worktree) is not bool or not self.clean_worktree:
            raise LifecycleError("execution requires an exact true clean-worktree attestation")


@dataclass(frozen=True)
class Decision:
    accepted: bool
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise LifecycleError("Decision.accepted must be an exact bool")
        if not isinstance(self.payload, Mapping):
            raise LifecycleError("Decision.payload must be a mapping")
        canonical_bytes(dict(self.payload))


@dataclass(frozen=True)
class RuntimeCallbacks:
    prebundle: Callable[["RunContext"], Mapping[str, Any]] | None
    development: Callable[["RunContext"], Decision]
    assessment: Callable[["RunContext"], Decision]


@dataclass(frozen=True)
class BundleOutcome:
    state: BundleState
    manifest: Mapping[str, Any]
    primary_error: Mapping[str, str] | None
    secondary_errors: tuple[Mapping[str, str], ...]


def _error_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)[:65536]}


def _lock_payload(
    experiment_id: str,
    producer: str,
    attestation: ExecutionAttestation,
) -> dict[str, Any]:
    return {
        "schema_version": LOCK_VERSION,
        "experiment_id": experiment_id,
        "producer_id": producer,
        "attempt": attestation.attempt,
        "commit": attestation.commit,
        "command": attestation.command,
        "host": attestation.host,
        "device": attestation.device,
        "clean_worktree_attested": attestation.clean_worktree,
        "acquired_at_utc": datetime.now(timezone.utc),
        "immutable": True,
    }


def acquire_execution_lock(
    root: SafeRoot | Path,
    payload: Mapping[str, Any],
    *,
    ops: FileOps | None = None,
) -> None:
    """Validate then create the immutable O_EXCL execution lock."""

    selected_ops = ops or FileOps()
    owns_root = not isinstance(root, SafeRoot)
    safe = root if isinstance(root, SafeRoot) else pin_existing_root(root, selected_ops)
    store = AtomicArtifactStore(safe, selected_ops)
    data = canonical_bytes(dict(payload))
    try:
        validate_lock(strict_json_loads(data))
    except ContractError as error:
        if owns_root:
            safe.close()
        raise LifecycleError(f"proposed lock schema is invalid: {error}") from error
    try:
        _exclusive_durable_write(selected_ops, safe.root, LOCK_NAME, data, 0o444)
        selected_ops.fsync_directory(safe.root)
        safe.verify()
    except FileExistsError as error:
        try:
            existing_data = store.read_bytes(LOCK_NAME)
            existing = strict_json_loads(existing_data)
            validate_lock(existing)
            if canonical_bytes_from_parsed(existing) != existing_data:
                raise ContractError("lock bytes are not canonical")
            same = all(
                existing[key] == strict_json_loads(data)[key]
                for key in ("experiment_id", "producer_id", "attempt", "commit")
            )
            classification = "same-attempt" if same else "different-attempt"
        except Exception:
            classification = "malformed"
        raise ExistingLockError(classification) from error
    finally:
        if owns_root:
            safe.close()


class RunContext:
    """Callback API; access records are committed before expensive work."""

    def __init__(self, runtime: "ExperimentRuntime", store: AtomicArtifactStore) -> None:
        self._runtime = runtime
        self.store = store

    @property
    def cache_root(self) -> Path:
        return self._runtime.cache.root

    def before_expensive(
        self,
        operation: str,
        *,
        kind: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if type(operation) is not str or not operation or len(operation) > 4096:
            raise LifecycleError("operation must be a bounded non-empty string")
        if kind not in {"solver", "label", "backend"}:
            raise LifecycleError(f"unsupported expensive-operation kind: {kind}")
        detail_value = dict(details or {})
        canonical_bytes(detail_value)
        self._runtime._counters["expensive_operation_count"] += 1
        if kind == "label":
            self._runtime._counters["label_access_count"] += 1
        self._runtime._write_counter(f"before-{kind}-{operation}")
        self._runtime._write_access(
            {
                "operation": operation,
                "kind": kind,
                "details": detail_value,
                "recorded_before_operation": True,
            }
        )

    def write_json(self, relative: str | Path, value: Any) -> dict[str, Any]:
        return self.store.write_json(relative, value)

    def write_blob(self, relative: str | Path, data: bytes) -> dict[str, Any]:
        if type(data) is not bytes:
            raise LifecycleError("blob data must be exact bytes")
        return self.store.write_blob(relative, data)

    def write_transcript(self, relative: str | Path, blob: bytes) -> dict[str, Any]:
        return self.write_blob(relative, blob)


def _paths_overlap(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(os.path.realpath(left.absolute()))
    right_text = os.path.normcase(os.path.realpath(right.absolute()))
    try:
        common = os.path.commonpath((left_text, right_text))
    except ValueError:
        return False
    return common in (left_text, right_text)


class ExperimentRuntime:
    """Own preflight, locking, callbacks, cache cleanup, and finalization."""

    def __init__(
        self,
        *,
        experiment_id: str,
        result_root: Path,
        cache_root: Path,
        attestation: ExecutionAttestation,
        producer: Any,
        source_root: Path,
        root_policy: RootPolicy | None = None,
        ops: FileOps | None = None,
    ) -> None:
        if type(experiment_id) is not str or not experiment_id or len(experiment_id) > 4096:
            raise LifecycleError("experiment_id must be a bounded non-empty string")
        attestation.validate()
        self.experiment_id = experiment_id
        self.result_root = result_root.absolute()
        self.attestation = attestation
        self.producer = producer_id(producer, source_root)
        self.root_policy = root_policy or RootPolicy()
        self.ops = ops or FileOps()
        self.cache = ManagedCache(cache_root.absolute(), experiment_id, self.ops)
        if _paths_overlap(self.result_root, self.cache.root):
            raise LifecycleError("working cache and result roots overlap")
        self.store: AtomicArtifactStore | None = None
        self._sequence = 0
        self._counter_sequence = 0
        self._access_sequence = 0
        self._events: list[dict[str, Any]] = []
        self._counters = {
            "attempt_count": 1,
            "prebundle_access_count": 0,
            "development_access_count": 0,
            "assessment_access_count": 0,
            "expensive_operation_count": 0,
            "label_access_count": 0,
        }

    def _write_transition(self, name: str, details: Mapping[str, Any] | None = None) -> None:
        if self.store is None:
            raise LifecycleError("artifact store is unavailable")
        self._sequence += 1
        event = {
            "schema_version": TRANSITION_VERSION,
            "sequence": self._sequence,
            "transition": name,
            "recorded_at_utc": datetime.now(timezone.utc),
            "details": dict(details or {}),
        }
        self.store.write_json(f"transitions/{self._sequence:04d}-{name}.json", event)
        self._events.append(event)

    def _write_counter(self, reason: str) -> None:
        if self.store is None:
            raise LifecycleError("artifact store is unavailable")
        self._counter_sequence += 1
        self.store.write_json(
            f"counters/{self._counter_sequence:04d}.json",
            {
                "schema_version": COUNTER_VERSION,
                "sequence": self._counter_sequence,
                "recorded_at_utc": datetime.now(timezone.utc),
                "reason": reason,
                "counts": dict(self._counters),
            },
        )

    def _write_access(self, value: Mapping[str, Any]) -> None:
        if self.store is None:
            raise LifecycleError("artifact store is unavailable")
        self._access_sequence += 1
        self.store.write_json(
            f"access/{self._access_sequence:04d}.json",
            {
                "schema_version": ACCESS_VERSION,
                "sequence": self._access_sequence,
                "recorded_at_utc": datetime.now(timezone.utc),
                "operation": value["operation"],
                "kind": value["kind"],
                "details": dict(value.get("details", {})),
                "recorded_before_operation": value["recorded_before_operation"],
            },
        )

    def _before_phase(self, phase: str) -> None:
        self._counters[f"{phase}_access_count"] += 1
        self._write_counter(f"before-{phase}")
        self._write_access(
            {
                "operation": phase,
                "kind": "phase",
                "details": {},
                "recorded_before_operation": True,
            }
        )
        self._write_transition(f"{phase}-started")

    def _evaluate(
        self,
        callbacks: RuntimeCallbacks,
        context: RunContext,
    ) -> tuple[BundleState, Mapping[str, Any], BaseException | None]:
        self._before_phase("prebundle")
        try:
            payload = callbacks.prebundle(context) if callbacks.prebundle else {}
            if not isinstance(payload, Mapping):
                raise TypeError("prebundle callback must return a mapping")
            self.store.write_json(  # type: ignore[union-attr]
                "phases/prebundle.json",
                dict(payload),
            )
            self._write_transition("prebundle-completed")
        except Exception as error:
            self._write_transition("prebundle-failed", {"error": _error_record(error)})
            return BundleState.PREBUNDLE_FAILURE, {}, error

        self._before_phase("development")
        try:
            development = callbacks.development(context)
            if type(development) is not Decision:
                raise TypeError("development callback must return exact Decision")
            self.store.write_json(  # type: ignore[union-attr]
                "phases/development.json",
                development,
            )
            if not development.accepted:
                self._write_transition("development-rejected")
                return BundleState.DEVELOPMENT_REJECTION, development.payload, None
            self._write_transition("development-accepted")
        except Exception as error:
            self._write_transition(
                "runtime-failed",
                {"phase": "development", "error": _error_record(error)},
            )
            return BundleState.RUNTIME_FAILURE, {}, error

        self._before_phase("assessment")
        try:
            assessment = callbacks.assessment(context)
            if type(assessment) is not Decision:
                raise TypeError("assessment callback must return exact Decision")
            self.store.write_json("phases/assessment.json", assessment)  # type: ignore[union-attr]
            if not assessment.accepted:
                self._write_transition("assessment-rejected")
                return BundleState.ASSESSMENT_REJECTION, assessment.payload, None
            self._write_transition("assessment-accepted")
            return BundleState.ACCEPTED_RESULT, assessment.payload, None
        except Exception as error:
            self._write_transition(
                "runtime-failed",
                {"phase": "assessment", "error": _error_record(error)},
            )
            return BundleState.RUNTIME_FAILURE, {}, error

    def run(self, callbacks: RuntimeCallbacks) -> BundleOutcome:
        if not callable(callbacks.development) or not callable(callbacks.assessment):
            raise LifecycleError("development and assessment callbacks must be callable")
        safe = preflight_result_root(
            self.result_root,
            policy=self.root_policy,
            ops=self.ops,
        )
        lock = _lock_payload(self.experiment_id, self.producer, self.attestation)
        try:
            acquire_execution_lock(safe, lock, ops=self.ops)
        except BaseException:
            safe.close()
            raise
        try:
            # This outer scope starts immediately after immutable lock creation
            # and closes every pinned handle even for BaseException/process-death
            # simulations which deliberately bypass terminal finalization.
            return self._run_locked(callbacks, safe, lock)
        finally:
            self.cache.close()
            safe.close()

    def _run_locked(
        self,
        callbacks: RuntimeCallbacks,
        safe: SafeRoot,
        lock: Mapping[str, Any],
    ) -> BundleOutcome:
        try:
            self.store = AtomicArtifactStore(safe, self.ops)
            context = RunContext(self, self.store)
            self._write_transition("lock-acquired", {"producer_id": self.producer})
            self._write_counter("lock-acquired")
            cache_prepared = False
            cache_state = self.cache.prepare()
            cache_prepared = True
            if self.cache.safe_root is None or identities_overlap(
                safe.root, self.cache.safe_root.root
            ):
                raise LifecycleError("pinned cache and result identities overlap")
            self._write_transition("cache-prepared", {"state": cache_state})
            state, payload, primary = self._evaluate(callbacks, context)
        except Exception as error:
            state, payload, primary = BundleState.PREBUNDLE_FAILURE, {}, error
        finally:
            cleanup_error: BaseException | None = None
            if locals().get("cache_prepared", False):
                try:
                    self.cache.cleanup(protected=safe)
                except BaseException as error:
                    cleanup_error = error
            else:
                self.cache.close()
        return self._complete_locked_attempt(
            state=state,
            payload=payload,
            primary=primary,
            cleanup_error=cleanup_error,
            lock=lock,
        )

    def _complete_locked_attempt(
        self,
        *,
        state: BundleState,
        payload: Mapping[str, Any],
        primary: BaseException | None,
        cleanup_error: BaseException | None,
        lock: Mapping[str, Any],
    ) -> BundleOutcome:
        cleanup_was_primary = cleanup_error is not None and primary is None
        if cleanup_was_primary:
            state, primary = BundleState.RUNTIME_FAILURE, cleanup_error
            cleanup_error = None
        secondary = () if cleanup_error is None else (_error_record(cleanup_error),)
        if self.store is None:
            raise LifecycleError("lock acquired but artifact store initialization failed")
        transition_names = {event["transition"] for event in self._events}
        if state is BundleState.PREBUNDLE_FAILURE and "prebundle-failed" not in transition_names:
            self._write_transition(
                "prebundle-failed",
                {"error": None if primary is None else _error_record(primary)},
            )
        if state is BundleState.RUNTIME_FAILURE and not (
            {"runtime-failed", "cleanup-failed"} & transition_names
        ):
            self._write_transition(
                "runtime-failed",
                {
                    "phase": "lifecycle",
                    "error": None if primary is None else _error_record(primary),
                },
            )
        if cleanup_was_primary and primary is not None:
            self._write_transition("cleanup-failed", {"error": _error_record(primary)})
        elif secondary:
            self._write_transition("cleanup-failed", {"error": secondary[0]})
        if secondary:
            self.store.write_json("secondary-failures/0001-cache-cleanup.json", secondary[0])
        primary_record = None if primary is None else _error_record(primary)
        terminal = {
            "schema_version": TERMINAL_VERSION,
            "state": state.value,
            "payload": dict(payload),
            "primary_error": primary_record,
            "secondary_errors": list(secondary),
            "counts": dict(self._counters),
        }
        self.store.write_json("terminal.json", terminal)
        self._write_transition("terminal", {"state": state.value})
        manifest = self._finalize_manifest(state, terminal, lock)
        return BundleOutcome(state, manifest, primary_record, secondary)

    def _finalize_manifest(
        self,
        state: BundleState,
        terminal: Mapping[str, Any],
        lock: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.store is None:
            raise LifecycleError("artifact store is unavailable")
        allowed_directories = self.store.declared_directories | set(
            self.root_policy.approved_directories
        )
        artifacts = _inventory(
            self.store,
            self.root_policy.approved_placeholders,
            allowed_directories,
        )
        identity = self.store.safe_root.identity
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "experiment_id": self.experiment_id,
            "state": state.value,
            "manifest_is_sole_completion_marker": True,
            "canonicalization": CANONICALIZATION_ID,
            "lock_byte_sha256": hashlib.sha256(canonical_bytes(lock)).hexdigest(),
            "transition_log_sha256": hashlib.sha256(canonical_bytes(self._events)).hexdigest(),
            "terminal_byte_sha256": hashlib.sha256(canonical_bytes(terminal)).hexdigest(),
            "artifact_count": len(artifacts),
            "required_directories": sorted(allowed_directories),
            "root_identity": {
                "platform": identity.platform,
                "volume": identity.volume,
                "file_id": identity.file_id,
                "final_path_sha256": hashlib.sha256(
                    identity.final_path.encode("utf-8")
                ).hexdigest(),
            },
            "durability": {
                **platform_durability_contract(),
                "directory_flush_supported": self.store.safe_root.directory_flush_supported,
            },
            "artifacts": artifacts,
        }
        candidate = canonical_bytes(manifest)
        sealed = self.store.seal_files(
            entry["path"] for entry in artifacts if entry["type"] == "file"
        )
        try:
            # Validate the complete candidate through an invisible override.
            # Every existing file remains deny-write/delete sealed on Windows
            # (and advisory-locked on POSIX) until publication completes.
            validate_bundle(
                self.store.safe_root,
                approved_placeholders=self.root_policy.approved_placeholders,
                manifest_override=candidate,
            )
            self.store.write_manifest_bytes(candidate)
        finally:
            for descriptor in sealed:
                os.close(descriptor)
        # Publishing is the final semantic operation. There is deliberately no
        # post-publication validator which could create a new rejection path.
        return manifest


def _inventory(
    store: AtomicArtifactStore,
    placeholders: Mapping[str, bytes],
    allowed_directories: set[str],
) -> list[dict[str, Any]]:
    entries = scan_tree(store.safe_root, store.ops)
    entry_paths = [name for name, _kind_name in entries]
    duplicate_paths = sorted(
        path for path, count in Counter(entry_paths).items() if count > 1
    )
    if duplicate_paths:
        raise FilesystemSafetyError(
            f"duplicate inventory path: {duplicate_paths[0]}"
        )
    entries = sorted(entries, key=lambda item: item[0])
    names = {name for name, _kind_name in entries}
    _validate_sidecar_bijection(store, entries, placeholders)
    result: list[dict[str, Any]] = []
    for relative, kind in entries:
        if relative == MANIFEST_NAME:
            continue
        if relative.endswith(TEMP_SUFFIX):
            raise FilesystemSafetyError(f"stale temporary entry: {relative}")
        if kind not in {"file", "directory"}:
            raise FilesystemSafetyError(f"unsafe inventory entry ({kind}): {relative}")
        if kind == "directory":
            if relative not in allowed_directories:
                raise FilesystemSafetyError(f"undeclared directory: {relative}")
            has_child = any(
                name.startswith(relative + "/") for name in names if name != relative
            )
            if not has_child and relative not in set():
                raise FilesystemSafetyError(f"empty directory is not allowed: {relative}")
            result.append(
                {
                    "path": relative,
                    "type": "directory",
                    "contract": "required-directory",
                }
            )
            continue
        data = store.read_bytes(relative)
        entry: dict[str, Any] = {
            "path": relative,
            "type": "file",
            "bytes": len(data),
            "byte_sha256": hashlib.sha256(data).hexdigest(),
        }
        if relative == LOCK_NAME:
            entry["contract"] = "immutable-lock"
        elif relative in placeholders and data == placeholders[relative]:
            entry["contract"] = "approved-placeholder"
        elif relative.endswith(SIDECAR_SUFFIX):
            entry["contract"] = "hash-sidecar-metadata"
        else:
            path = Path(relative)
            sidecar = path.with_name(path.name + SIDECAR_SUFFIX).as_posix()
            if sidecar not in names:
                raise FilesystemSafetyError(f"uncontracted artifact: {relative}")
            contract = verify_pair(store, relative)
            entry.update(
                {
                    "contract": "hash-sidecar",
                    "sidecar": sidecar,
                    "semantic_sha256": contract["semantic_sha256"],
                }
            )
        try:
            parsed = strict_json_loads(data)
            if canonical_bytes_from_parsed(parsed) == data:
                validate_encoded_value(parsed, relative)
                entry["canonical_json_sha256"] = hashlib.sha256(data).hexdigest()
        except Exception:
            if entry["contract"] in {"immutable-lock", "hash-sidecar-metadata"} or entry.get(
                "semantic_sha256"
            ) is not None:
                raise
        result.append(entry)
    return result


def _validate_sidecar_bijection(
    store: AtomicArtifactStore,
    entries: list[tuple[str, str]],
    placeholders: Mapping[str, bytes],
) -> None:
    files = {name for name, kind in entries if kind == "file"}
    sidecars = {name for name in files if name.endswith(SIDECAR_SUFFIX)}
    artifacts = {
        name
        for name in files
        if name not in {LOCK_NAME, MANIFEST_NAME}
        and name not in placeholders
        and not name.endswith(SIDECAR_SUFFIX)
    }
    if any(
        sidecar[: -len(SIDECAR_SUFFIX)].endswith(SIDECAR_SUFFIX)
        for sidecar in sidecars
    ):
        raise FilesystemSafetyError("sidecar-of-sidecar entry is forbidden")

    declared_references: dict[str, list[str]] = {}
    parsed_contracts: dict[str, Any] = {}
    for sidecar in sorted(sidecars):
        contract = strict_json_loads(store.read_bytes(sidecar))
        parsed_contracts[sidecar] = contract
        if type(contract) is dict and type(contract.get("artifact")) is str:
            declared_references.setdefault(contract["artifact"], []).append(sidecar)
    duplicates = {
        artifact: references
        for artifact, references in declared_references.items()
        if len(references) != 1
    }
    if duplicates:
        raise FilesystemSafetyError(
            f"duplicate sidecar artifact reference: {sorted(duplicates)[0]}"
        )

    referenced: set[str] = set()
    for sidecar in sorted(sidecars):
        artifact = sidecar[: -len(SIDECAR_SUFFIX)]
        if artifact not in artifacts:
            raise FilesystemSafetyError(f"orphan sidecar: {sidecar}")
        contract = parsed_contracts[sidecar]
        if type(contract) is not dict or contract.get("artifact") != artifact:
            raise FilesystemSafetyError(
                f"mismatched sidecar contract: {sidecar}"
            )
        verify_pair(store, artifact)
        referenced.add(artifact)
    missing = artifacts - referenced
    if missing:
        raise FilesystemSafetyError(
            f"artifact has no sidecar: {sorted(missing)[0]}"
        )


def diagnose_bundle(
    root: SafeRoot | Path,
    *,
    approved_placeholders: Mapping[str, bytes] | None = None,
) -> tuple[str, ...]:
    if isinstance(root, Path) and not os.path.lexists(root):
        return ("missing-root",)
    owns_root = not isinstance(root, SafeRoot)
    safe = root if isinstance(root, SafeRoot) else pin_existing_root(root)
    try:
        entries = scan_tree(safe)
        kinds = dict(entries)
        issues: list[str] = []
        for name, kind in entries:
            if kind not in {"file", "directory"}:
                issues.append(f"unsafe-entry:{kind}:{name}")
            if name.endswith(TEMP_SUFFIX):
                issues.append("stale-temp")
        files = {name for name, kind in entries if kind == "file"}
        placeholders = dict(approved_placeholders or {})
        artifacts = {
            name
            for name in files
            if name not in (LOCK_NAME, MANIFEST_NAME)
            and not name.endswith(SIDECAR_SUFFIX)
            and not name.endswith(TEMP_SUFFIX)
            and not (name in placeholders)
        }
        sidecars = {name for name in files if name.endswith(SIDECAR_SUFFIX)}
        for artifact in artifacts:
            path = Path(artifact)
            sidecar = path.with_name(path.name + SIDECAR_SUFFIX).as_posix()
            if sidecar not in sidecars:
                issues.append(f"partial-pair:{artifact}")
        for sidecar in sidecars:
            base = sidecar[: -len(SIDECAR_SUFFIX)]
            if base not in artifacts:
                issues.append(f"orphan-sidecar:{sidecar}")
        directories = {name for name, kind in entries if kind == "directory"}
        for directory in directories:
            if not any(name.startswith(directory + "/") for name in kinds if name != directory):
                issues.append(f"empty-directory:{directory}")
        if LOCK_NAME not in files:
            issues.append("missing-lock")
        if MANIFEST_NAME not in files:
            issues.append("incomplete-no-manifest")
        return tuple(sorted(set(issues)))
    finally:
        if owns_root:
            safe.close()


def _validate_manifest(value: Any) -> dict[str, Any]:
    keys = {
        "schema_version",
        "experiment_id",
        "state",
        "manifest_is_sole_completion_marker",
        "canonicalization",
        "lock_byte_sha256",
        "transition_log_sha256",
        "terminal_byte_sha256",
        "artifact_count",
        "required_directories",
        "root_identity",
        "durability",
        "artifacts",
    }
    manifest = exact_mapping(value, keys, "manifest")
    if manifest["schema_version"] != MANIFEST_VERSION:
        raise ContractError("unsupported manifest schema")
    nonempty_string(manifest["experiment_id"], "manifest.experiment_id")
    if manifest["state"] not in {state.value for state in BundleState}:
        raise ContractError("manifest state is unknown")
    if not exact_bool(
        manifest["manifest_is_sole_completion_marker"],
        "manifest.completion",
    ):
        raise ContractError("manifest completion marker must be true")
    if manifest["canonicalization"] != CANONICALIZATION_ID:
        raise ContractError("unsupported manifest canonicalization")
    for key in ("lock_byte_sha256", "transition_log_sha256", "terminal_byte_sha256"):
        sha256_string(manifest[key], f"manifest.{key}")
    bounded_int(manifest["artifact_count"], "manifest.artifact_count", 1)
    directories = manifest["required_directories"]
    if (
        type(directories) is not list
        or directories != sorted(set(directories))
        or any(type(item) is not str for item in directories)
    ):
        raise ContractError("manifest.required_directories must be sorted and unique")
    for item in directories:
        relative_path(item)
    identity = exact_mapping(
        manifest["root_identity"],
        {"platform", "volume", "file_id", "final_path_sha256"},
        "manifest.root_identity",
    )
    nonempty_string(identity["platform"], "manifest.root_identity.platform")
    bounded_int(identity["volume"], "manifest.root_identity.volume")
    bounded_int(identity["file_id"], "manifest.root_identity.file_id")
    sha256_string(identity["final_path_sha256"], "manifest.root_identity.final_path_sha256")
    expected_durability = set(platform_durability_contract()) | {
        "directory_flush_supported"
    }
    durability = exact_mapping(
        manifest["durability"],
        expected_durability,
        "manifest.durability",
    )
    for key in expected_durability - {"directory_flush_supported"}:
        nonempty_string(durability[key], f"manifest.durability.{key}")
        if durability[key] != platform_durability_contract()[key]:
            raise ContractError(f"manifest.durability.{key} is not the platform contract")
    exact_bool(
        durability["directory_flush_supported"],
        "manifest.durability.directory_flush_supported",
    )
    if type(manifest["artifacts"]) is not list:
        raise ContractError("manifest.artifacts must be a list")
    artifact_paths: list[str] = []
    directory_paths: list[str] = []
    for index, entry in enumerate(manifest["artifacts"]):
        name = f"manifest.artifacts[{index}]"
        if type(entry) is not dict or entry.get("type") not in {"file", "directory"}:
            raise ContractError(f"{name} has invalid entry type")
        relative_path(entry.get("path", ""))
        artifact_paths.append(entry["path"])
        if entry["type"] == "directory":
            exact_mapping(entry, {"path", "type", "contract"}, name)
            if entry["contract"] != "required-directory":
                raise ContractError(f"{name} has invalid directory contract")
            directory_paths.append(entry["path"])
        else:
            required = {"path", "type", "bytes", "byte_sha256", "contract"}
            optional = {"canonical_json_sha256", "semantic_sha256", "sidecar"}
            if not required.issubset(entry) or not set(entry).issubset(required | optional):
                raise ContractError(f"{name} has invalid file keys")
            bounded_int(entry["bytes"], f"{name}.bytes")
            sha256_string(entry["byte_sha256"], f"{name}.byte_sha256")
            if "canonical_json_sha256" in entry:
                sha256_string(entry["canonical_json_sha256"], f"{name}.canonical")
            if "semantic_sha256" in entry and entry["semantic_sha256"] is not None:
                sha256_string(entry["semantic_sha256"], f"{name}.semantic")
            if "sidecar" in entry:
                relative_path(entry["sidecar"])
            if entry["contract"] not in {
                "immutable-lock",
                "approved-placeholder",
                "hash-sidecar",
                "hash-sidecar-metadata",
            }:
                raise ContractError(f"{name} has invalid file contract")
            if (entry["contract"] == "hash-sidecar") != ("sidecar" in entry):
                raise ContractError(f"{name} sidecar link does not match contract")
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ContractError("manifest artifacts contain duplicate paths")
    if artifact_paths != sorted(artifact_paths):
        raise ContractError(
            "manifest artifacts are unsorted; paths must be globally path-sorted"
        )
    if directory_paths != directories:
        raise ContractError("required directories do not match directory inventory")
    if manifest["artifact_count"] != len(manifest["artifacts"]):
        raise ContractError("manifest artifact_count mismatch")
    return manifest


def _validate_transition(value: Any) -> dict[str, Any]:
    event = exact_mapping(
        value,
        {"schema_version", "sequence", "transition", "recorded_at_utc", "details"},
        "transition",
    )
    if event["schema_version"] != TRANSITION_VERSION:
        raise ContractError("unsupported transition schema")
    bounded_int(event["sequence"], "transition.sequence", 1)
    nonempty_string(event["transition"], "transition.transition")
    validate_utc_tag(event["recorded_at_utc"], "transition.recorded_at_utc")
    if type(event["details"]) is not dict:
        raise ContractError("transition.details must be an object")
    validate_encoded_value(event["details"], "transition.details")
    return event


def _validate_counter(value: Any) -> dict[str, Any]:
    row = exact_mapping(
        value,
        {"schema_version", "sequence", "recorded_at_utc", "reason", "counts"},
        "counter",
    )
    if row["schema_version"] != COUNTER_VERSION:
        raise ContractError("unsupported counter schema")
    bounded_int(row["sequence"], "counter.sequence", 1)
    validate_utc_tag(row["recorded_at_utc"], "counter.recorded_at_utc")
    nonempty_string(row["reason"], "counter.reason")
    validate_counts(row["counts"])
    return row


def _validate_access(value: Any) -> dict[str, Any]:
    row = exact_mapping(
        value,
        {
            "schema_version",
            "sequence",
            "recorded_at_utc",
            "operation",
            "kind",
            "details",
            "recorded_before_operation",
        },
        "access",
    )
    if row["schema_version"] != ACCESS_VERSION:
        raise ContractError("unsupported access schema")
    bounded_int(row["sequence"], "access.sequence", 1)
    validate_utc_tag(row["recorded_at_utc"], "access.recorded_at_utc")
    nonempty_string(row["operation"], "access.operation")
    if row["kind"] not in {"phase", "solver", "label", "backend"}:
        raise ContractError("access.kind is invalid")
    if type(row["details"]) is not dict:
        raise ContractError("access.details must be an object")
    validate_encoded_value(row["details"], "access.details")
    if not exact_bool(row["recorded_before_operation"], "access.recorded"):
        raise ContractError("access must be recorded before operation")
    return row


def _validate_decision_record(value: Any, name: str) -> None:
    record = exact_mapping(
        value,
        {"__cft_type__", "class", "fields"},
        name,
    )
    if record["__cft_type__"] != "dataclass":
        raise ContractError(f"{name} must be a tagged dataclass")
    class_name = nonempty_string(record["class"], f"{name}.class")
    if not class_name.endswith(".Decision"):
        raise ContractError(f"{name} must encode Decision")
    fields = exact_mapping(record["fields"], {"accepted", "payload"}, f"{name}.fields")
    exact_bool(fields["accepted"], f"{name}.accepted")
    if type(fields["payload"]) is not dict:
        raise ContractError(f"{name}.payload must be an object")
    validate_encoded_value(fields["payload"], f"{name}.payload")


def validate_bundle(
    root: SafeRoot | Path,
    *,
    approved_placeholders: Mapping[str, bytes] | None = None,
    manifest_override: bytes | None = None,
) -> Mapping[str, Any]:
    """Validate a visible manifest or an invisible pre-publication candidate."""

    owns_root = not isinstance(root, SafeRoot)
    safe = root if isinstance(root, SafeRoot) else pin_existing_root(root)
    store = AtomicArtifactStore(safe)
    try:
        safe.verify()
        visible_manifest = MANIFEST_NAME in {
            name for name, kind in scan_tree(safe) if kind == "file"
        }
        if manifest_override is not None:
            if visible_manifest:
                raise LifecycleError("candidate validation requires no visible manifest")
            manifest_data = manifest_override
        else:
            if not visible_manifest:
                raise LifecycleError("bundle has no completion manifest")
            manifest_data = store.read_bytes(MANIFEST_NAME)
        manifest_value = strict_json_loads(manifest_data)
        if canonical_bytes_from_parsed(manifest_value) != manifest_data:
            raise LifecycleError("manifest is not canonical")
        manifest = _validate_manifest(manifest_value)
        state = BundleState(manifest["state"])
        allowed_directories = set(manifest["required_directories"])
        expected = _inventory(
            store,
            dict(approved_placeholders or {}),
            allowed_directories,
        )
        if manifest["artifacts"] != expected:
            raise LifecycleError("manifest inventory is incomplete or changed")
        identity = safe.identity
        expected_identity = {
            "platform": identity.platform,
            "volume": identity.volume,
            "file_id": identity.file_id,
            "final_path_sha256": hashlib.sha256(
                identity.final_path.encode("utf-8")
            ).hexdigest(),
        }
        if manifest["root_identity"] != expected_identity:
            raise LifecycleError("manifest root identity does not match pinned root")
        if (
            manifest["durability"]["directory_flush_supported"]
            is not safe.directory_flush_supported
        ):
            raise LifecycleError("manifest directory-flush claim does not match platform")
        lock_data = store.read_bytes(LOCK_NAME)
        lock = strict_json_loads(lock_data)
        validate_lock(lock)
        if canonical_bytes_from_parsed(lock) != lock_data:
            raise LifecycleError("execution lock is not canonical")
        if lock["experiment_id"] != manifest["experiment_id"]:
            raise LifecycleError("lock and manifest experiment identities differ")
        if manifest["lock_byte_sha256"] != hashlib.sha256(lock_data).hexdigest():
            raise LifecycleError("execution-lock identity mismatch")
        terminal_data = store.read_bytes("terminal.json")
        terminal = validate_terminal(strict_json_loads(terminal_data))
        if canonical_bytes_from_parsed(terminal) != terminal_data:
            raise LifecycleError("terminal is not canonical")
        if terminal["state"] != state.value:
            raise LifecycleError("terminal and manifest states differ")

        event_paths = sorted(
            entry["path"]
            for entry in expected
            if entry["type"] == "file"
            and entry["path"].startswith("transitions/")
            and not entry["path"].endswith(SIDECAR_SUFFIX)
        )
        events = [
            _validate_transition(strict_json_loads(store.read_bytes(item)))
            for item in event_paths
        ]
        if [item["sequence"] for item in events] != list(range(1, len(events) + 1)):
            raise LifecycleError("transition sequence is not contiguous")
        if not events or events[0]["transition"] != "lock-acquired":
            raise LifecycleError("transition log does not start at lock acquisition")
        if events[-1]["transition"] != "terminal":
            raise LifecycleError("transition log has no terminal transition")
        if manifest["transition_log_sha256"] != hashlib.sha256(
            canonical_bytes_from_parsed(events)
        ).hexdigest():
            raise LifecycleError("transition log identity mismatch")
        transitions = [item["transition"] for item in events]
        invalid = [
            pair
            for pair in zip(transitions[:-1], transitions[1:], strict=True)
            if pair not in EVENT_TRANSITION_PAIRS
        ]
        if invalid:
            raise LifecycleError(f"invalid state-machine transition: {invalid[0]}")
        if "prebundle-completed" in transitions:
            prebundle = strict_json_loads(store.read_bytes("phases/prebundle.json"))
            if type(prebundle) is not dict or "__cft_type__" in prebundle:
                raise LifecycleError("prebundle phase record must be a plain object")
            validate_encoded_value(prebundle, "phases.prebundle")
        if {
            "development-accepted",
            "development-rejected",
        } & set(transitions):
            _validate_decision_record(
                strict_json_loads(store.read_bytes("phases/development.json")),
                "phases.development",
            )
        if {"assessment-accepted", "assessment-rejected"} & set(transitions):
            _validate_decision_record(
                strict_json_loads(store.read_bytes("phases/assessment.json")),
                "phases.assessment",
            )

        counter_paths = sorted(
            entry["path"]
            for entry in expected
            if entry["type"] == "file"
            and entry["path"].startswith("counters/")
            and not entry["path"].endswith(SIDECAR_SUFFIX)
        )
        counters = [
            _validate_counter(strict_json_loads(store.read_bytes(item)))
            for item in counter_paths
        ]
        if [item["sequence"] for item in counters] != list(range(1, len(counters) + 1)):
            raise LifecycleError("counter sequence is not contiguous")
        previous: dict[str, int] = {}
        for row in counters:
            current = row["counts"]
            if any(current[key] < value for key, value in previous.items()):
                raise LifecycleError("a lifecycle counter decreased")
            previous = current
        if not counters or counters[-1]["counts"] != terminal["counts"]:
            raise LifecycleError("terminal counters differ from last atomic counter")

        access_paths = sorted(
            entry["path"]
            for entry in expected
            if entry["type"] == "file"
            and entry["path"].startswith("access/")
            and not entry["path"].endswith(SIDECAR_SUFFIX)
        )
        accesses = [
            _validate_access(strict_json_loads(store.read_bytes(item)))
            for item in access_paths
        ]
        if [item["sequence"] for item in accesses] != list(range(1, len(accesses) + 1)):
            raise LifecycleError("access sequence is not contiguous")
        counts = terminal["counts"]
        if counts["label_access_count"] != sum(item["kind"] == "label" for item in accesses):
            raise LifecycleError("label count differs from access records")
        for phase in ("prebundle", "development", "assessment"):
            observed = sum(
                item["kind"] == "phase" and item["operation"] == phase
                for item in accesses
            )
            if counts[f"{phase}_access_count"] != observed:
                raise LifecycleError(f"{phase} count differs from access records")
        assessment_started = "assessment-started" in transitions
        if bool(counts["assessment_access_count"]) != assessment_started:
            raise LifecycleError("assessment counter/transition mismatch")
        if "development-rejected" in transitions and assessment_started:
            raise LifecycleError("assessment ran after development rejection")
        required_predecessors = {
            BundleState.PREBUNDLE_FAILURE: {"prebundle-failed", "cleanup-failed"},
            BundleState.RUNTIME_FAILURE: {"runtime-failed", "cleanup-failed"},
            BundleState.DEVELOPMENT_REJECTION: {"development-rejected"},
            BundleState.ASSESSMENT_REJECTION: {"assessment-rejected"},
            BundleState.ACCEPTED_RESULT: {"assessment-accepted"},
        }
        if transitions[-2] not in required_predecessors[state]:
            raise LifecycleError("terminal predecessor does not match bundle state")
        if manifest["terminal_byte_sha256"] != hashlib.sha256(terminal_data).hexdigest():
            raise LifecycleError("terminal identity mismatch")
        safe.verify()
        return manifest
    except (ContractError, FilesystemSafetyError) as error:
        raise LifecycleError(f"bundle contract rejected: {error}") from error
    finally:
        if owns_root:
            safe.close()
