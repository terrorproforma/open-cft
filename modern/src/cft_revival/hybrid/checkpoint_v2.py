"""``hybrid-checkpoint-v2``: the L2 v2 simulation state, hash-bound to its inputs.

The v1 checkpoint (``checkpoint.py``) serialises a handful of Cartesian macroparticles as JSON.
v2 stores the structure-of-arrays state of ``HybridL2Simulation`` (ions, wall surface charge,
potential, per-cell electron fluid, neutral inventory, ledgers, RNG state) in a deterministic npz
whose byte hash is bound by a canonical JSON metadata file that also binds

* the configuration identity (``content_hash(config.to_dict())``),
* the cell partition identity,
* the cross-section payload hash and the rate-table hash,
* the field: ``field_sha256`` (bitwise, platform-bound) AND ``field_source_sha256`` (the
  platform-independent P2 bundle / grid binding introduced by the PIC in ``0ac8d9b8``) with an
  anchor copy of the node arrays, so a resume elsewhere is admitted only under the PIC's
  declared numerical-replay tolerance (``pic2d.artifacts.verify_field_identity``),
* the code identity of the hybrid package and the runtime / platform fingerprint.

Loading fails closed on any mismatch.  As for v1, the digests detect corruption and drift; they
are not a signature.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..orbit_mc.artifacts import content_hash
from ..pic2d import artifacts as pic_artifacts
from ..pic2d.fields import MagneticFieldMap
from ..pic2d.models import ParticleArrays
from ..pic2d.neutrals import NEUTRAL_LEDGER_KEYS, NeutralState
from .ions import IonPopulation
from .l2 import CUMULATIVE_KEYS, HybridL2Simulation, HybridL2State
from .models import HybridValidationError

CHECKPOINT_V2_SCHEMA = "hybrid-checkpoint-v2"


def hybrid_code_identity() -> str:
    """SHA-256 over the hybrid package sources (name + bytes, sorted)."""

    root = Path(__file__).resolve().parent
    digest = sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rng_state_to_json(state: dict[str, Any]) -> dict[str, Any]:
    """numpy BitGenerator state with every integer as a decimal string (canonical JSON is int64-bound)."""

    def convert(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(state)


def rng_state_from_json(state: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(state)


def save_checkpoint_v2(directory: str | Path, name: str, simulation: HybridL2Simulation) -> tuple[Path, Path]:
    """Write ``<name>.npz``, ``<name>.field.npz`` (anchor) and ``<name>.json``; returns (json, npz) paths."""

    directory = Path(directory)
    st = simulation.state
    p = st.ions.particles
    arrays: dict[str, np.ndarray] = {
        "ions_r_m": p.r_m, "ions_z_m": p.z_m, "ions_vr_m_per_s": p.vr_m_per_s, "ions_vt_m_per_s": p.vt_m_per_s, "ions_vz_m_per_s": p.vz_m_per_s,
        "surface_charge_c": st.surface_charge_c, "phi_v": st.phi_v, "log_reference": st.log_reference,
        "electron_count": st.electron_count, "electron_energy_j": st.electron_energy_j, "neutral": st.neutral.to_array(),
        "cumulative": np.array([st.cumulative[key] for key in CUMULATIVE_KEYS], dtype=np.float64),
        "scalars": np.array([st.birth_carry], dtype=np.float64),
    }
    if simulation._previous_density is not None:
        arrays["previous_density"] = simulation._previous_density
    if simulation._previous_cell_phi is not None:
        arrays["previous_cell_phi"] = simulation._previous_cell_phi
    npz_path = directory / f"{name}.npz"
    npz_sha = pic_artifacts.write_npz(npz_path, arrays)
    field = simulation.field
    anchor_path = directory / f"{name}.field.npz"
    anchor_sha = pic_artifacts.write_npz(anchor_path, {"b_r_t": field.b_r_t, "b_z_t": field.b_z_t})
    metadata = {
        "schema_version": CHECKPOINT_V2_SCHEMA,
        "step": int(st.step), "time_s": float(st.time_s), "ion_count": p.count, "cell_count": int(st.electron_count.size),
        "arrays_file": npz_path.name, "arrays_sha256": npz_sha, "cumulative_keys": list(CUMULATIVE_KEYS),
        "neutral_keys": ["density_per_m3", *NEUTRAL_LEDGER_KEYS],
        "config_sha256": content_hash(simulation.config.to_dict()),
        "partition_sha256": content_hash(simulation.partition.to_dict()),
        "cross_section_sha256": simulation.cross_sections.payload_sha256,
        "rate_table_sha256": simulation.rates.sha256(),
        "field_sha256": field.sha256, "field_source_sha256": field.source_sha256,
        "field_anchor_file": anchor_path.name, "field_anchor_sha256": anchor_sha,
        "field_replay_policy": "bitwise when field_sha256 matches; otherwise the live map must share field_source_sha256 and lie within "
                               "the PIC's declared tolerance of the anchor arrays (pic2d.fields.compare_field_arrays)",
        "rng_state": rng_state_to_json(st.rng.bit_generator.state),
        "series_records": len(simulation.series),
        "runtime": pic_artifacts.runtime_identity() | {"hybrid_code_sha256": hybrid_code_identity()},
        "staggering": "ions x^n, v^(n-1/2); phi^(n-1) retained as a warm start; electron fluid counts/energies at n",
    }
    json_path = directory / f"{name}.json"
    pic_artifacts.write_canonical_json(json_path, metadata)
    return json_path, npz_path


def load_checkpoint_v2(json_path: str | Path, simulation: HybridL2Simulation, *, require_same_code: bool = True) -> dict[str, Any]:
    """Restore the state of ``simulation`` from a v2 checkpoint, failing closed on any identity mismatch."""

    json_path = Path(json_path)
    metadata = pic_artifacts.read_canonical_json(json_path)
    if metadata.get("schema_version") != CHECKPOINT_V2_SCHEMA:
        raise HybridValidationError("unsupported hybrid checkpoint schema")
    if metadata.get("config_sha256") != content_hash(simulation.config.to_dict()):
        raise HybridValidationError("checkpoint configuration identity differs")
    if metadata.get("partition_sha256") != content_hash(simulation.partition.to_dict()):
        raise HybridValidationError("checkpoint cell partition differs")
    if metadata.get("cross_section_sha256") != simulation.cross_sections.payload_sha256:
        raise HybridValidationError("checkpoint cross-section identity differs")
    if metadata.get("rate_table_sha256") != simulation.rates.sha256():
        raise HybridValidationError("checkpoint rate-table identity differs")
    field: MagneticFieldMap = simulation.field
    field_report = pic_artifacts.verify_field_identity(metadata, json_path, field_sha256=field.sha256, field=field)
    if require_same_code and (metadata.get("runtime") or {}).get("hybrid_code_sha256") != hybrid_code_identity():
        raise HybridValidationError("checkpoint hybrid code identity differs")
    if list(metadata.get("cumulative_keys", [])) != list(CUMULATIVE_KEYS):
        raise HybridValidationError("checkpoint ledger keys differ")
    arrays = pic_artifacts.read_npz(json_path.with_name(metadata["arrays_file"]), expected_sha256=metadata["arrays_sha256"])
    ions = ParticleArrays(arrays["ions_r_m"], arrays["ions_z_m"], arrays["ions_vr_m_per_s"], arrays["ions_vt_m_per_s"], arrays["ions_vz_m_per_s"])
    if ions.count != metadata["ion_count"]:
        raise HybridValidationError("checkpoint ion count differs from metadata")
    shape = simulation.config.grid.node_shape
    for key in ("surface_charge_c", "phi_v"):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise HybridValidationError(f"checkpoint {key} does not match the grid or is nonfinite")
    k = simulation.partition.cell_count
    for key in ("log_reference", "electron_count", "electron_energy_j"):
        if arrays[key].shape != (k,) or not np.isfinite(arrays[key]).all():
            raise HybridValidationError(f"checkpoint {key} does not match the partition")
    cumulative = {key: float(value) for key, value in zip(CUMULATIVE_KEYS, arrays["cumulative"], strict=True)}
    rng = np.random.default_rng(simulation.config.seed)
    rng.bit_generator.state = rng_state_from_json(metadata["rng_state"])
    simulation.state = HybridL2State(
        int(metadata["step"]), float(metadata["time_s"]), IonPopulation(simulation.species, ions),
        np.asarray(arrays["surface_charge_c"], dtype=np.float64), np.asarray(arrays["phi_v"], dtype=np.float64),
        np.asarray(arrays["log_reference"], dtype=np.float64), np.asarray(arrays["electron_count"], dtype=np.float64),
        np.asarray(arrays["electron_energy_j"], dtype=np.float64), NeutralState.from_array(arrays["neutral"]),
        float(arrays["scalars"][0]), cumulative, rng,
    )
    simulation._previous_density = np.asarray(arrays["previous_density"], dtype=np.float64) if "previous_density" in arrays else None
    simulation._previous_cell_phi = np.asarray(arrays["previous_cell_phi"], dtype=np.float64) if "previous_cell_phi" in arrays else None
    simulation._pending = None
    return {"field": field_report, "step": int(metadata["step"]), "metadata": metadata}


def checkpoint_metadata(json_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


__all__ = ["CHECKPOINT_V2_SCHEMA", "checkpoint_metadata", "hybrid_code_identity", "load_checkpoint_v2", "rng_state_from_json",
           "rng_state_to_json", "save_checkpoint_v2"]
