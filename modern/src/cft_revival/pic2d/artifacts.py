"""Canonical JSON / npz artifacts, SHA-256 sidecars, checkpoints and provenance."""

from __future__ import annotations

import ctypes
from functools import lru_cache
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
from .fields import FIELD_REPLAY_ATOL_OVER_MAX_B, FIELD_REPLAY_RTOL, MagneticFieldMap, compare_field_arrays
from .models import PIC2DValidationError, ParticleArrays
from .neutrals import NEUTRAL_LEDGER_KEYS, NeutralState
from .simulation import CUMULATIVE_KEYS, PIC2DConfig, SimulationState

SIDECAR_SCHEMA = "cft.pic2d.artifact-sidecar.v1"
CHECKPOINT_SCHEMA = "cft.pic2d.checkpoint.v1"
PLATFORM_FINGERPRINT_SCHEMA = "cft.pic2d.platform-fingerprint.v1"
# The keys of ``platform_fingerprint()`` that determine floating-point round-off of the CPU code paths (OS / libm
# and the numpy wheel's compiler, the SIMD dispatch targets in use, the BLAS build and its runtime kernel set);
# everything else in the record (CPU model string, Python patch level, OS release) is informational.
PLATFORM_FINGERPRINT_KEYS = ("os", "machine", "libc", "numpy", "numpy_c_compiler", "simd_baseline", "simd_dispatch_enabled", "blas")


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


def _cpu_model() -> str | None:
    """Marketing name of the CPU (informational; /proc/cpuinfo on Linux, the registry on Windows, else platform)."""

    fallback = platform.processor() or None
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Windows":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:  # pragma: no cover - best effort only
        return fallback
    return fallback


def _openblas_runtime_corename() -> str | None:
    """The OpenBLAS kernel set selected at runtime (DYNAMIC_ARCH), read from numpy's bundled library; None if unavailable."""

    libs = Path(np.__file__).resolve().parent.parent / "numpy.libs"
    candidates = sorted(libs.glob("*openblas*")) if libs.is_dir() else []
    for candidate in candidates:
        try:
            library = ctypes.CDLL(str(candidate))
        except OSError:  # pragma: no cover - depends on the wheel layout
            continue
        for symbol in ("scipy_openblas_get_corename64_", "openblas_get_corename64_", "scipy_openblas_get_corename", "openblas_get_corename"):
            function = getattr(library, symbol, None)
            if function is not None:
                function.restype = ctypes.c_char_p
                value = function()
                return value.decode("ascii", "replace") if value else None
    return None


@lru_cache(maxsize=1)
def platform_fingerprint() -> dict[str, Any]:
    """The CPU-side floating-point platform of this process, with a SHA-256 fingerprint over its determinants.

    Two processes with the same ``fingerprint_sha256`` run the same numpy wheel (compiler, FMA contraction), the
    same SIMD dispatch targets, the same BLAS build and runtime kernel set on the same OS / libm: CPU-side
    reductions and the P2 field sampling replay bitwise between them.  Anywhere else only numerical replay under
    declared tolerances is expected (the repo's cross-platform policy).
    """

    config = np.show_config(mode="dicts") or {}
    dependencies = config.get("Build Dependencies") or {}
    blas = dependencies.get("blas") or {}
    compilers = config.get("Compilers") or {}
    c_compiler = compilers.get("c") or {}
    try:
        from numpy._core import _multiarray_umath as umath

        baseline = list(getattr(umath, "__cpu_baseline__", []))
        dispatch = list(getattr(umath, "__cpu_dispatch__", []))
        features = getattr(umath, "__cpu_features__", {}) or {}
        enabled = [name for name in dispatch if features.get(name)]
    except Exception:  # pragma: no cover - numpy layout change
        baseline, enabled = [], []
    libc = platform.libc_ver()
    record: dict[str, Any] = {
        "schema": PLATFORM_FINGERPRINT_SCHEMA,
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "libc": [libc[0], libc[1]],
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "numpy_c_compiler": {"name": c_compiler.get("name"), "version": c_compiler.get("version")},
        "simd_baseline": baseline,
        "simd_dispatch_enabled": enabled,
        "blas": {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "build_configuration": blas.get("openblas configuration"),
            "runtime_corename": _openblas_runtime_corename(),
        },
        "cpu_model": _cpu_model(),
    }
    record["fingerprint_sha256"] = content_hash({key: record[key] for key in PLATFORM_FINGERPRINT_KEYS})
    return record


_GPU_IDENTITY: dict[str, Any] | None = None


def gpu_identity() -> dict[str, Any] | None:
    """Name / architecture / UUID of the CUDA device Warp is using and the driver and toolkit versions.

    Read from an ALREADY initialised Warp runtime only (never forces ``wp.init()`` from a CPU run, never calls
    ``nvidia-smi`` on the stepping thread - attempt-7 lesson); None when Warp is absent or uninitialised.  The
    first successful reading is cached for the process.
    """

    global _GPU_IDENTITY
    if _GPU_IDENTITY is not None:
        return _GPU_IDENTITY
    try:
        import warp as wp

        try:
            from warp._src import context  # warp >= 1.10 layout
        except ImportError:  # pragma: no cover - older warp
            from warp import context  # type: ignore[no-redef]
        runtime = getattr(context, "runtime", None)
        if runtime is None:
            return None
        device = wp.get_cuda_device() if wp.is_cuda_available() else None
        record: dict[str, Any] = {
            "driver_version": list(getattr(runtime, "driver_version", None) or []) or None,
            "toolkit_version": list(getattr(runtime, "toolkit_version", None) or []) or None,
        }
        if device is not None:
            record |= {
                "name": device.name, "arch": int(device.arch), "uuid": getattr(device, "uuid", None),
                "pci_bus_id": getattr(device, "pci_bus_id", None), "total_memory_bytes": int(device.total_memory),
            }
        _GPU_IDENTITY = record
        return record
    except Exception:  # pragma: no cover - optional dependency
        return None


def runtime_identity() -> dict[str, Any]:
    """Interpreter, code and platform identity recorded with every checkpoint and summary.

    ``platform_fingerprint`` (CPU side: OS, numpy build, SIMD dispatch, BLAS kernel, CPU model) and ``gpu``
    (device, driver, toolkit) make a record honest about WHERE it was produced: a checkpoint written under one
    fingerprint replays bitwise only under the same one; elsewhere the numerical-replay gates apply.
    """

    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "code_sha256": code_identity(),
        "platform_fingerprint": platform_fingerprint(),
    }
    try:
        import warp as wp

        record["warp"] = getattr(wp, "__version__", None)
    except Exception:  # pragma: no cover - optional dependency
        record["warp"] = None
    record["gpu"] = gpu_identity()
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
    field: MagneticFieldMap | None = None,
) -> tuple[Path, Path]:
    """Write ``<name>.npz`` (arrays) and ``<name>.json`` (metadata binding the arrays' hash).

    With ``field`` (the live node map) the metadata also binds the platform-independent
    ``field_source_sha256`` and an anchor copy of the node arrays is written to ``<name>.field.npz`` (its byte
    hash in the metadata), so a resume on another CPU / BLAS / OS can verify the SAME field source and gate
    the re-sampled map against the recorded one under the declared tolerance instead of refusing on a
    last-digit content-hash difference.  ``<name>.npz`` itself is unchanged by this option.
    """

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
    # v1.4: optional tallies (sensitivity hooks) beyond the fixed ledger, e.g. "anomalous"
    extra_keys = sorted(key for key in state.cumulative if key not in CUMULATIVE_KEYS)
    if extra_keys:
        arrays["cumulative_extra"] = np.array([state.cumulative[key] for key in extra_keys], dtype=np.float64)
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
        **({"cumulative_extra_keys": extra_keys} if extra_keys else {}),
        "arrays_file": npz_path.name,
        "arrays_sha256": npz_sha,
        "config_sha256": config_identity(config),
        "field_sha256": field_sha256,
        "cross_section_sha256": cross_section_sha256,
        "backend": backend,
        "runtime": runtime_identity(),
        "staggering": "positions x^n; velocities v^(n-1/2); phi^(n-1) retained only as a warm start",
    }
    if field is not None:
        if field.sha256 != field_sha256:
            raise PIC2DValidationError("field_sha256 does not belong to the field map passed as the anchor")
        anchor_path = directory / f"{name}.field.npz"
        metadata |= {
            "field_source_sha256": field.source_sha256,
            "field_anchor_file": anchor_path.name,
            "field_anchor_sha256": write_npz(anchor_path, {"b_r_t": field.b_r_t, "b_z_t": field.b_z_t}),
            "field_replay_policy": "bitwise when field_sha256 matches; otherwise the live map must share field_source_sha256 and lie "
                                   f"within rtol {FIELD_REPLAY_RTOL:g} / atol {FIELD_REPLAY_ATOL_OVER_MAX_B:g} x max|B| of the anchor arrays",
        }
    json_path = directory / f"{name}.json"
    write_canonical_json(json_path, metadata)
    return json_path, npz_path


def verify_field_identity(
    metadata: Mapping[str, Any],
    json_path: Path,
    *,
    field_sha256: str,
    field: MagneticFieldMap | None,
) -> dict[str, Any]:
    """The checkpoint's field binding against the live map (fail-closed); returns the replay record.

    * legacy metadata (no ``field_source_sha256``) or no live map given: the content hash must match exactly;
    * otherwise the platform-independent source identity must match, then the content hash decides between a
      ``bitwise`` replay and a ``numerical`` one, which is admitted only when the live arrays lie within the
      declared tolerance of the recorded anchor arrays.
    """

    recorded = metadata.get("field_sha256")
    if field is None or metadata.get("field_source_sha256") is None:
        if recorded != field_sha256:
            raise PIC2DValidationError("checkpoint field identity differs")
        return {"mode": "bitwise", "field_sha256": recorded, "basis": "content hash (legacy binding: no source identity / anchor recorded)"}
    if field.sha256 != field_sha256:
        raise PIC2DValidationError("field_sha256 does not belong to the live field map")
    if metadata["field_source_sha256"] != field.source_sha256:
        raise PIC2DValidationError("checkpoint field source identity differs (different P2 checkpoint bundle, grid, scale or extension)")
    if recorded == field_sha256:
        return {"mode": "bitwise", "field_sha256": recorded, "field_source_sha256": field.source_sha256, "basis": "content hash equal"}
    anchor_path = json_path.with_name(str(metadata.get("field_anchor_file", "")))
    if not metadata.get("field_anchor_file") or not anchor_path.is_file():
        raise PIC2DValidationError("checkpoint field identity differs and no anchor arrays are recorded for a numerical replay check")
    anchor = read_npz(anchor_path, expected_sha256=metadata["field_anchor_sha256"])
    comparison = compare_field_arrays(field, anchor["b_r_t"], anchor["b_z_t"])
    if not comparison["within_tolerance"]:
        raise PIC2DValidationError(
            "checkpoint field differs from the live map beyond the declared cross-platform tolerance "
            f"(max |dB| {comparison['max_abs_diff_t']:.3e} T, max relative {comparison['max_rel_diff']:.3e}; "
            f"rtol {comparison['rtol']:g}, atol {comparison['atol_t']:.3e} T)"
        )
    return {
        "mode": "numerical", "anchor_field_sha256": recorded, "live_field_sha256": field_sha256, "field_source_sha256": field.source_sha256,
        "basis": "same source identity; live arrays within the declared tolerance of the recorded anchor (not a bitwise replay)",
        "comparison": comparison,
        "anchor_platform_fingerprint": ((metadata.get("runtime") or {}).get("platform_fingerprint") or {}).get("fingerprint_sha256"),
        "live_platform_fingerprint": platform_fingerprint()["fingerprint_sha256"],
    }


def load_checkpoint(
    json_path: str | Path,
    config: PIC2DConfig,
    *,
    field_sha256: str,
    cross_section_sha256: str | None,
    require_same_code: bool = True,
    field: MagneticFieldMap | None = None,
    identity_report: dict[str, Any] | None = None,
) -> SimulationState:
    """Reload a checkpoint, failing closed on any identity or hash mismatch.

    With ``field`` the field binding is checked by ``verify_field_identity`` (source identity + bitwise or
    tolerance-gated numerical replay); without it the content hash must match exactly.  ``identity_report``
    (a dict) receives the replay record under ``"field"`` for the caller's provenance.
    """

    json_path = Path(json_path)
    metadata = read_canonical_json(json_path)
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA:
        raise PIC2DValidationError("unsupported checkpoint schema")
    if metadata.get("config_sha256") != config_identity(config):
        raise PIC2DValidationError("checkpoint configuration identity differs")
    field_report = verify_field_identity(metadata, json_path, field_sha256=field_sha256, field=field)
    if identity_report is not None:
        identity_report["field"] = field_report
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
    extra_keys = metadata.get("cumulative_extra_keys")
    if extra_keys:
        if "cumulative_extra" not in arrays or len(extra_keys) != int(arrays["cumulative_extra"].size):
            raise PIC2DValidationError("checkpoint extra tallies do not match their keys")
        cumulative |= {key: float(value) for key, value in zip(extra_keys, arrays["cumulative_extra"], strict=True)}
    neutral = None
    if metadata.get("neutral_keys") is not None or "neutral" in arrays:
        accepted = (["density_per_m3", *NEUTRAL_LEDGER_KEYS], ["density_per_m3", *NEUTRAL_LEDGER_KEYS[:4]])  # v1.4 / v1.3 layouts
        if metadata.get("neutral_keys") not in accepted or "neutral" not in arrays:
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
    "PLATFORM_FINGERPRINT_KEYS",
    "PLATFORM_FINGERPRINT_SCHEMA",
    "SIDECAR_SCHEMA",
    "code_identity",
    "config_identity",
    "gpu_identity",
    "load_checkpoint",
    "platform_fingerprint",
    "read_canonical_json",
    "read_npz",
    "runtime_identity",
    "save_checkpoint",
    "verify_field_identity",
    "write_canonical_json",
    "write_npz",
]
