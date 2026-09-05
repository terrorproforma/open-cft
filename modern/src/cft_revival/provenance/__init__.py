"""Bind sealed source digests to the commit that sealed them, not to the live tree.

A preregistered experiment seals SHA-256 digests of its own code and of the shared packages
it imports (``experiment_code_sha256``, ``dependency_source_sha256`` ...). Those digests are
evidence about the tree AT THE EXECUTION COMMIT. Recomputing them from the live worktree proves
the seal only until the next commit to a shared dependency; after the terminal bundle exists the
honest check is (a) recompute the digest from the frozen commit's blobs and assert equality with
the sealed value, and (b) RECORD the live tree's digest with a drift flag instead of asserting it.
This package is the shared plumbing for (a) and (b). It is deliberately not part of
``cft_revival.experiment_runtime``: several sealed dependency scopes glob that package, so a new
module there would itself be live-tree drift.
"""

from .frozen_sources import (
    FrozenBlobError,
    FrozenCommitError,
    SealedScope,
    blob_exists,
    frozen_scope_report,
    path_bytes_sha256,
    read_blobs,
    resolve_commit,
    verify_sealed_scopes,
)

__all__ = [
    "FrozenBlobError",
    "FrozenCommitError",
    "SealedScope",
    "blob_exists",
    "frozen_scope_report",
    "path_bytes_sha256",
    "read_blobs",
    "resolve_commit",
    "verify_sealed_scopes",
]
