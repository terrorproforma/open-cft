"""Root-confined atomic artifact serialization for L0 surrogate v3."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable

from cft_revival.surrogates import ExactGP

Replace = Callable[[str | bytes | Path, str | bytes | Path], object]


class ArtifactWriteError(RuntimeError):
    """An artifact could not be written atomically."""


class AtomicArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path(self, relative: str | Path) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ArtifactWriteError("artifact path must be root-relative without traversal")
        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ArtifactWriteError("artifact path escapes its declared root")
        return resolved

    def write_bytes(
        self,
        relative: str | Path,
        payload: bytes,
        *,
        replace: Replace = os.replace,
    ) -> Path:
        target = self.path(relative)
        temporary: Path | None = None
        handle = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            )
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            replace(temporary, target)
            temporary = None
            return target
        except Exception as error:
            if handle is not None:
                handle.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise ArtifactWriteError(f"atomic write failed for {relative}") from error

    def write_text(
        self,
        relative: str | Path,
        payload: str,
        *,
        replace: Replace = os.replace,
    ) -> Path:
        return self.write_bytes(relative, payload.encode("utf-8"), replace=replace)

    def write_json(
        self,
        relative: str | Path,
        payload: object,
        *,
        replace: Replace = os.replace,
    ) -> Path:
        serialized = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        return self.write_text(relative, serialized, replace=replace)

    def write_model(self, relative: str | Path, model: ExactGP) -> Path:
        target = self.write_text(relative, model.dumps())
        reloaded = ExactGP.load(target)
        if reloaded.model_hash != model.model_hash:
            target.unlink(missing_ok=True)
            raise ArtifactWriteError("serialized model hash changed on reload")
        return target

    def temporary_files(self) -> tuple[Path, ...]:
        if not self.root.exists():
            return ()
        return tuple(self.root.rglob(".*.tmp"))
