"""Closed production validators for every runtime-owned record."""

from __future__ import annotations

import base64
import math
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Mapping

from .canonical import CANONICALIZATION_ID, TYPE_KEY, WINDOWS_RESERVED_NAMES

LOCK_VERSION = "cft-revival.experiment-execution-lock/1.0.0"
SIDECAR_VERSION = "cft-revival.experiment-artifact-sidecar/1.0.0"
TERMINAL_VERSION = "cft-revival.experiment-terminal/1.0.0"
MANIFEST_VERSION = "cft-revival.experiment-manifest/1.0.0"
TRANSITION_VERSION = "cft-revival.experiment-transition/1.0.0"
COUNTER_VERSION = "cft-revival.experiment-counter/1.0.0"
ACCESS_VERSION = "cft-revival.experiment-access/1.0.0"
TERMINAL_STATES = frozenset(
    {
        "prebundle_failure",
        "runtime_failure",
        "development_rejection",
        "assessment_rejection",
        "accepted_result",
    }
)
COUNTER_KEYS = frozenset(
    {
        "attempt_count",
        "prebundle_access_count",
        "development_access_count",
        "assessment_access_count",
        "expensive_operation_count",
        "label_access_count",
    }
)
HEX_64 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
QUALIFIED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
PRODUCER = re.compile(r"[^:\\]+:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


class ContractError(ValueError):
    """A runtime-owned payload violates its closed schema."""


def exact_mapping(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        actual = set(value) if type(value) is dict else type(value).__name__
        raise ContractError(f"{name} requires exact keys {sorted(keys)}; got {actual}")
    return value


def exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{name} must be an exact bool")
    return value


def bounded_int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > 2**63 - 1:
        raise ContractError(f"{name} must be an integer in [{minimum}, 2^63-1]")
    return value


def nonempty_string(value: Any, name: str, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ContractError(f"{name} must be a non-empty bounded string")
    return value


def sha256_string(value: Any, name: str) -> str:
    text = nonempty_string(value, name, 64)
    if not HEX_64.fullmatch(text):
        raise ContractError(f"{name} must be lowercase SHA-256")
    return text


def validate_utc_tag(value: Any, name: str) -> None:
    item = exact_mapping(value, {TYPE_KEY, "value"}, name)
    if item[TYPE_KEY] != "aware-utc-datetime":
        raise ContractError(f"{name} must use the aware UTC datetime tag")
    text = nonempty_string(item["value"], f"{name}.value", 32)
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:"
        r"[0-9]{2}:[0-9]{2}\.[0-9]{6}Z",
        text,
    ):
        raise ContractError(f"{name} must use canonical UTC microsecond form")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(f"{name} is not ISO-8601") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ContractError(f"{name} is not UTC")


def _validate_relative_posix(text: str, name: str) -> None:
    path = PurePosixPath(text)
    if (
        not text
        or text.startswith("/")
        or "\\" in text
        or ":" in text
        or text.endswith("/")
        or any(
            part in ("", ".", "..")
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
            or any(ord(character) < 32 for character in part)
            for part in path.parts
        )
        or path.as_posix() != text
    ):
        raise ContractError(f"{name} is not a canonical relative POSIX path")


def validate_encoded_value(value: Any, name: str = "value") -> None:
    if value is None or type(value) in (bool, str):
        return
    if type(value) is int:
        bounded_int(value, name, -(2**63))
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ContractError(f"{name} must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            validate_encoded_value(item, f"{name}[{index}]")
        return
    if type(value) is not dict:
        raise ContractError(f"{name} has unsupported JSON type")
    if TYPE_KEY not in value:
        for key, item in value.items():
            if type(key) is not str:
                raise ContractError(f"{name} keys must be strings")
            validate_encoded_value(item, f"{name}.{key}")
        return
    tag = value.get(TYPE_KEY)
    if tag == "aware-utc-datetime":
        validate_utc_tag(value, name)
    elif tag == "relative-posix-path":
        tagged = exact_mapping(value, {TYPE_KEY, "value"}, name)
        _validate_relative_posix(nonempty_string(tagged["value"], name), name)
    elif tag == "bytes-base64":
        tagged = exact_mapping(value, {TYPE_KEY, "value"}, name)
        text = tagged["value"]
        if type(text) is not str or len(text) > 16 * 1024 * 1024:
            raise ContractError(f"{name} must contain bounded base64 text")
        try:
            base64.b64decode(text, validate=True)
        except ValueError as error:
            raise ContractError(f"{name} has invalid base64") from error
    elif tag == "tuple":
        tagged = exact_mapping(value, {TYPE_KEY, "items"}, name)
        if type(tagged["items"]) is not list:
            raise ContractError(f"{name}.items must be a list")
        for index, item in enumerate(tagged["items"]):
            validate_encoded_value(item, f"{name}.items[{index}]")
    elif tag == "enum":
        tagged = exact_mapping(value, {TYPE_KEY, "class", "value"}, name)
        if not QUALIFIED.fullmatch(nonempty_string(tagged["class"], f"{name}.class")):
            raise ContractError(f"{name}.class is not qualified")
        validate_encoded_value(tagged["value"], f"{name}.value")
    elif tag == "dataclass":
        tagged = exact_mapping(value, {TYPE_KEY, "class", "fields"}, name)
        if not QUALIFIED.fullmatch(nonempty_string(tagged["class"], f"{name}.class")):
            raise ContractError(f"{name}.class is not qualified")
        if type(tagged["fields"]) is not dict:
            raise ContractError(f"{name}.fields must be an object")
        for key, item in tagged["fields"].items():
            if not QUALIFIED.fullmatch(key):
                raise ContractError(f"{name}.fields contains an invalid name")
            validate_encoded_value(item, f"{name}.fields.{key}")
    else:
        raise ContractError(f"{name} has unknown type tag {tag!r}")


def validate_lock(value: Any) -> dict[str, Any]:
    keys = {
        "schema_version",
        "experiment_id",
        "producer_id",
        "attempt",
        "commit",
        "command",
        "host",
        "device",
        "clean_worktree_attested",
        "acquired_at_utc",
        "immutable",
    }
    lock = exact_mapping(value, keys, "lock")
    if lock["schema_version"] != LOCK_VERSION:
        raise ContractError("unsupported lock schema")
    nonempty_string(lock["experiment_id"], "lock.experiment_id")
    producer = nonempty_string(lock["producer_id"], "lock.producer_id")
    if not PRODUCER.fullmatch(producer) or "\\" in producer.split(":", 1)[0]:
        raise ContractError("lock producer_id is not relative-file:qualified-name")
    _validate_relative_posix(producer.split(":", 1)[0], "lock.producer_id.file")
    bounded_int(lock["attempt"], "lock.attempt", 1)
    commit = nonempty_string(lock["commit"], "lock.commit", 64)
    if not COMMIT.fullmatch(commit):
        raise ContractError("lock.commit is invalid")
    for key in ("command", "host", "device"):
        nonempty_string(lock[key], f"lock.{key}")
    if not exact_bool(lock["clean_worktree_attested"], "lock.clean"):
        raise ContractError("lock clean attestation must be true")
    if not exact_bool(lock["immutable"], "lock.immutable"):
        raise ContractError("lock immutable must be true")
    validate_utc_tag(lock["acquired_at_utc"], "lock.acquired_at_utc")
    return lock


def validate_counts(value: Any, name: str = "counts") -> dict[str, int]:
    counts = exact_mapping(value, set(COUNTER_KEYS), name)
    for key in COUNTER_KEYS:
        bounded_int(counts[key], f"{name}.{key}")
    if counts["attempt_count"] != 1:
        raise ContractError("attempt_count must equal one")
    return counts


def validate_error(value: Any, name: str) -> None:
    error = exact_mapping(value, {"type", "message"}, name)
    nonempty_string(error["type"], f"{name}.type", 256)
    if type(error["message"]) is not str or len(error["message"]) > 65536:
        raise ContractError(f"{name}.message must be a bounded string")


def validate_terminal(value: Any) -> dict[str, Any]:
    terminal = exact_mapping(
        value,
        {
            "schema_version",
            "state",
            "payload",
            "primary_error",
            "secondary_errors",
            "counts",
        },
        "terminal",
    )
    if terminal["schema_version"] != TERMINAL_VERSION:
        raise ContractError("unsupported terminal schema")
    if terminal["state"] not in TERMINAL_STATES:
        raise ContractError("unknown terminal state")
    if type(terminal["payload"]) is not dict:
        raise ContractError("terminal.payload must be an object")
    validate_encoded_value(terminal["payload"], "terminal.payload")
    if terminal["primary_error"] is not None:
        validate_error(terminal["primary_error"], "terminal.primary_error")
    if type(terminal["secondary_errors"]) is not list:
        raise ContractError("terminal.secondary_errors must be a list")
    for index, item in enumerate(terminal["secondary_errors"]):
        validate_error(item, f"terminal.secondary_errors[{index}]")
    validate_counts(terminal["counts"], "terminal.counts")
    failure = terminal["state"] in {"prebundle_failure", "runtime_failure"}
    if failure != (terminal["primary_error"] is not None):
        raise ContractError("terminal primary error does not match failure state")
    return terminal
