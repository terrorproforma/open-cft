"""Canonical JSON / npz artifacts, SHA-256 sidecars, checkpoints and provenance."""

from __future__ import annotations

from hashlib import sha256
import io
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import numpy as np

from ..orbit_mc.artifacts import canonical_bytes, content_hash
from .models import PIC2DValidationError, ParticleArrays
from .neutrals import NEUTRAL_LEDGER_KEYS, NeutralState
from .simulation import CUMULATIVE_KEYS, PIC2DConfig, SimulationState

SIDECAR_SCHEMA = "cft.pic2d.artifact-sidecar.v1"
CHECKPOINT_SCHEMA = "cft.pic2d.checkpoint.v1"


def code_identity() -> str:
    """SHA-256 over the package sources (name + bytes, sorted)."""

    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _write_sidecar(path: Path, data: bytes, *, semantic_sha256: str | None = None) -> str:
    digest = sha256(data).hexdigest()
    sidecar = {
        "schema_version": SIDECAR_SCHEMA,
        "artifact": path.name,
        "bytes": len(data),
        "byte_sha256": digest,
        "semantic_sha256": semantic_sha256 or digest,
    }
    _write_bytes_atomic(path.with_name(path.name + ".sha256.json"), canonical_bytes(sidecar) + b"\n")
    return digest


def write_canonical_json(path: str | Path, value: Mapping[str, Any]) -> str:
    """Write sorted, compact, finite JSON with a hash sidecar; return the byte SHA-256."""

    target = Path(path)
    data = canonical_bytes(value) + b"\n"
    _write_bytes_atomic(target, data)
    return _write_sidecar(target, data, semantic_sha256=content_hash(value))


def read_canonical_json(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    target = Path(path)
    data = target.read_bytes()
    digest = sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise PIC2DValidationError(f"{target.name}: byte SHA-256 mismatch")
    sidecar_path = target.with_name(target.name + ".sha256.json")
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("byte_sha256") != digest:
            raise PIC2DValidationError(f"{target.name}: sidecar SHA-256 mismatch")
    loaded = json.loads(data.decode("utf-8"))
    if canonical_bytes(loaded) + b"\n" != data:
        raise PIC2DValidationError(f"{target.name}: file is not canonical JSON")
    return loaded


def write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Write a deterministic (uncompressed, sorted-key) npz with a hash sidecar."""

    target = Path(path)
    buffer = io.BytesIO()
    ordered = {key: np.ascontiguousarray(arrays[key]) for key in sorted(arrays)}
    np.savez(buffer, **ordered)
    data = buffer.getvalue()
    _write_bytes_atomic(target, data)
    return _write_sidecar(target, data)


def read_npz(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    target = Path(path)
    data = target.read_bytes()
    digest = sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise PIC2DValidationError(f"{target.name}: byte SHA-256 mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def runtime_identity() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "code_sha256": code_identity(),
    }
    try:
        import warp as wp

        record["warp"] = getattr(wp, "__version__", None)
    except Exception:  # pragma: no cover - optional dependency
        record["warp"] = None
    return record


def config_identity(config: PIC2DConfig) -> str:
    return content_hash(config.to_dict())


def _particles_to_arrays(prefix: str, particles: ParticleArrays) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_r_m": particles.r_m,
        f"{prefix}_z_m": particles.z_m,
        f"{prefix}_vr_m_per_s": particles.vr_m_per_s,
        f"{prefix}_vt_m_per_s": particles.vt_m_per_s,
        f"{prefix}_vz_m_per_s": particles.vz_m_per_s,
    }


def _particles_from_arrays(prefix: str, arrays: Mapping[str, np.ndarray]) -> ParticleArrays:
    return ParticleArrays(
        arrays[f"{prefix}_r_m"], arrays[f"{prefix}_z_m"], arrays[f"{prefix}_vr_m_per_s"],
        arrays[f"{prefix}_vt_m_per_s"], arrays[f"{prefix}_vz_m_per_s"],
    )


def save_checkpoint(
    directory: str | Path,
    name: str,
    state: SimulationState,
    config: PIC2DConfig,
    *,
    field_sha256: str,
    cross_section_sha256: str | None,
    backend: str,
) -> tuple[Path, Path]:
    """Write ``<name>.npz`` (arrays) and ``<name>.json`` (metadata binding the arrays' hash)."""

    directory = Path(directory)
    arrays: dict[str, np.ndarray] = {
        **_particles_to_arrays("electrons", state.electrons),
        **_particles_to_arrays("ions", state.ions),
        "surface_charge_c": state.surface_charge_c,
        "phi_v": state.phi_v,
        "cumulative": np.array([state.cumulative[key] for key in CUMULATIVE_KEYS], dtype=np.float64),
    }
    if state.neutral is not None:
        # v1.3: n_g and the atom ledgers travel with the arrays (hash-bound like the particles)
        arrays["neutral"] = state.neutral.to_array()
    npz_path = directory / f"{name}.npz"
    npz_sha = write_npz(npz_path, arrays)
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "step": int(state.step),
        "time_s": float(state.time_s),
        "injection_carry": float(state.injection_carry),
        "electron_count": state.electrons.count,
        "ion_count": state.ions.count,
        "cumulative_keys": list(CUMULATIVE_KEYS),
        "neutral_keys": None if state.neutral is None else ["density_per_m3", *NEUTRAL_LEDGER_KEYS],
        "arrays_file": npz_path.name,
        "arrays_sha256": npz_sha,
        "config_sha256": config_identity(config),
        "field_sha256": field_sha256,
        "cross_section_sha256": cross_section_sha256,
        "backend": backend,
        "runtime": runtime_identity(),
        "staggering": "positions x^n; velocities v^(n-1/2); phi^(n-1) retained only as a warm start",
    }
    json_path = directory / f"{name}.json"
    write_canonical_json(json_path, metadata)
    return json_path, npz_path


def load_checkpoint(
    json_path: str | Path,
    config: PIC2DConfig,
    *,
    field_sha256: str,
    cross_section_sha256: str | None,
    require_same_code: bool = True,
) -> SimulationState:
    """Reload a checkpoint, failing closed on any identity or hash mismatch."""

    json_path = Path(json_path)
    metadata = read_canonical_json(json_path)
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA:
        raise PIC2DValidationError("unsupported checkpoint schema")
    if metadata.get("config_sha256") != config_identity(config):
        raise PIC2DValidationError("checkpoint configuration identity differs")
    if metadata.get("field_sha256") != field_sha256:
        raise PIC2DValidationError("checkpoint field identity differs")
    if metadata.get("cross_section_sha256") != cross_section_sha256:
        raise PIC2DValidationError("checkpoint cross-section identity differs")
    if require_same_code and metadata.get("runtime", {}).get("code_sha256") != code_identity():
        raise PIC2DValidationError("checkpoint code identity differs")
    if list(metadata.get("cumulative_keys", [])) != list(CUMULATIVE_KEYS):
        raise PIC2DValidationError("checkpoint cumulative ledger keys differ")
    arrays = read_npz(json_path.with_name(metadata["arrays_file"]), expected_sha256=metadata["arrays_sha256"])
    electrons = _particles_from_arrays("electrons", arrays)
    ions = _particles_from_arrays("ions", arrays)
    if electrons.count != metadata["electron_count"] or ions.count != metadata["ion_count"]:
        raise PIC2DValidationError("checkpoint particle counts differ from metadata")
    surface = np.asarray(arrays["surface_charge_c"], dtype=np.float64)
    phi = np.asarray(arrays["phi_v"], dtype=np.float64)
    if surface.shape != config.grid.node_shape or phi.shape != config.grid.node_shape:
        raise PIC2DValidationError("checkpoint node arrays do not match the grid")
    if not np.isfinite(surface).all() or not np.isfinite(phi).all():
        raise PIC2DValidationError("checkpoint node arrays are nonfinite")
    cumulative = {key: float(value) for key, value in zip(CUMULATIVE_KEYS, arrays["cumulative"], strict=True)}
    neutral = None
    if metadata.get("neutral_keys") is not None or "neutral" in arrays:
        if metadata.get("neutral_keys") != ["density_per_m3", *NEUTRAL_LEDGER_KEYS] or "neutral" not in arrays:
            raise PIC2DValidationError("checkpoint neutral inventory keys differ")
        neutral = NeutralState.from_array(arrays["neutral"])
    if (config.neutral_inventory is not None) != (neutral is not None):
        raise PIC2DValidationError("checkpoint neutral inventory presence does not match the configuration")
    return SimulationState(
        int(metadata["step"]), float(metadata["time_s"]), electrons, ions, surface, phi,
        float(metadata["injection_carry"]), cumulative, neutral,
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "SIDECAR_SCHEMA",
    "code_identity",
    "config_identity",
    "load_checkpoint",
    "read_canonical_json",
    "read_npz",
    "runtime_identity",
    "save_checkpoint",
    "write_canonical_json",
    "write_npz",
]
