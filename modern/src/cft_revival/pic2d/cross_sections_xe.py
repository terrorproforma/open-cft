"""Xenon collision-set selection (model v2.3.0, ``xe_collision_set_v2``).

This module extends the hash-bound cross-section handling of :mod:`cft_revival.pic2d.mcc` with

* the FOUR-LEVEL electron set ``spec/pic2d/xenon-cross-sections-v2.json`` (Biagi-v7.1 levels 8.315 /
  9.447 / 9.917 / 11.7 eV instead of the lumped 8.32 eV channel of v1; same database, same bytes) -
  loaded through :meth:`XenonCrossSections.from_file`, which accepts the v1 and v2 schemas;
* the Xe+ + Xe ion-neutral set ``spec/pic2d/xenon-ion-neutral-cross-sections-v1.json`` (Miller 2002
  charge exchange + Phelps isotropic momentum transfer), consumed by :mod:`cft_revival.pic2d.ion_mcc`;
* :class:`CollisionSetConfig`, the declaration that enters ``MCCConfig.to_dict`` and therefore
  ``config_sha256``: the set name, the electron file and payload hash, the process list with its
  thresholds, and the optional ion-neutral block.  A configuration without a collision set is the
  legacy v1 lumped set with collisionless ions and keeps its identity bitwise.

Identity policy: the payload hashes bound here are the ``integrity.payload_sha256`` of the spec files
(sha256 of the canonical JSON without the ``integrity`` block).  ``Simulation`` refuses a cross-section
object whose payload hash differs from the declaration (fail closed), so a protocol that names the set
cannot silently run other data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .ion_mcc import IonNeutralCrossSections, IonNeutralMCCConfig
from .mcc import SPEC_DIR, XenonCrossSections
from .models import PIC2DValidationError

XE_ELECTRON_SET_V2_FILE = "xenon-cross-sections-v2.json"
XE_ION_NEUTRAL_SET_V1_FILE = "xenon-ion-neutral-cross-sections-v1.json"
XE_COLLISION_SET_V2_NAME = "xe_collision_set_v2"
# payload sha256 of the spec files as built by spec/pic2d/build_xenon_cross_sections_v2.py and
# build_xenon_ion_neutral_cross_sections.py (the loaders recompute and compare; a rebuild that changes the
# payload must update these pins together with the model spec entry)
XE_ELECTRON_SET_V2_PAYLOAD_SHA256 = "9b39858afc4aa5e94e66c90f46772ca011674cb4c5ca93bb7c2930fec5698228"
XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256 = "6f259ba9abdf17c317dc67f535b25d23e14ab44f88667c4667392642bac079cb"


def spec_path(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else SPEC_DIR / path


@dataclass(frozen=True, slots=True)
class CollisionSetConfig:
    """Declared xenon collision set (enters ``config_sha256`` through ``MCCConfig.to_dict``)."""

    name: str
    electron_file: str
    electron_payload_sha256: str
    electron_processes: tuple[tuple[str, str, float], ...]      # (id, kind, threshold_ev) in table order
    ion_neutral: IonNeutralMCCConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise PIC2DValidationError("collision set name must be a non-empty string")
        if not isinstance(self.electron_payload_sha256, str) or len(self.electron_payload_sha256) != 64:
            raise PIC2DValidationError("electron_payload_sha256 must be a 64-hex sha256")
        if len(self.electron_processes) < 3:
            raise PIC2DValidationError("the electron process list needs elastic, >= 1 excitation level and ionization")
        object.__setattr__(self, "electron_processes", tuple((str(i), str(k), float(t)) for i, k, t in self.electron_processes))

    @classmethod
    def from_cross_sections(cls, name: str, electron_file: str, cross_sections: XenonCrossSections,
                            ion_neutral: IonNeutralMCCConfig | None = None) -> "CollisionSetConfig":
        return cls(name, electron_file, cross_sections.payload_sha256,
                   tuple((p.identifier, p.kind, p.threshold_ev) for p in cross_sections.processes), ion_neutral)

    @classmethod
    def xe_collision_set_v2(cls, *, ion_neutral: bool = True, **ion_kwargs: Any) -> "CollisionSetConfig":
        """The v2.3.0 production set: four-level electron set + Xe+ / Xe CEX and MEX (``ion_neutral=False`` = R3a only)."""

        electron = XenonCrossSections.from_file(spec_path(XE_ELECTRON_SET_V2_FILE))
        if electron.payload_sha256 != XE_ELECTRON_SET_V2_PAYLOAD_SHA256:
            raise PIC2DValidationError(f"{XE_ELECTRON_SET_V2_FILE}: payload sha256 {electron.payload_sha256} differs from the pinned set")
        ion: IonNeutralMCCConfig | None = None
        if ion_neutral:
            ion_set = IonNeutralCrossSections.from_file(spec_path(XE_ION_NEUTRAL_SET_V1_FILE))
            if ion_set.payload_sha256 != XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256:
                raise PIC2DValidationError(f"{XE_ION_NEUTRAL_SET_V1_FILE}: payload sha256 {ion_set.payload_sha256} differs from the pinned set")
            ion = IonNeutralMCCConfig(XE_ION_NEUTRAL_SET_V1_FILE, ion_set.payload_sha256,
                                      tuple((p.identifier, p.kind) for p in ion_set.processes), **ion_kwargs)
        return cls.from_cross_sections(XE_COLLISION_SET_V2_NAME, XE_ELECTRON_SET_V2_FILE, electron, ion)

    @classmethod
    def from_protocol(cls, block: Mapping[str, Any]) -> "CollisionSetConfig":
        """Protocol block ``operating_point.collision_set``: ``{"name": "xe_collision_set_v2", "ion_neutral": {...} | false}``.

        The hashes are NOT read from the protocol: the named set is loaded from the spec files and its recomputed payload
        hashes are what enter the identity, so a protocol cannot declare data it does not ship.  Optional ion-neutral
        keys (``energy_step_ev``, ``energy_max_ev``, ``fast_neutral_speed_threshold_factor``) are forwarded.
        """

        name = str(block.get("name", ""))
        if name != XE_COLLISION_SET_V2_NAME:
            raise PIC2DValidationError(f"unknown collision set {name!r} (known: {XE_COLLISION_SET_V2_NAME!r})")
        ion_block = block.get("ion_neutral", True)
        if ion_block is False or ion_block is None:
            return cls.xe_collision_set_v2(ion_neutral=False)
        kwargs = {} if ion_block is True else {k: v for k, v in dict(ion_block).items() if not k.endswith("_note")}
        return cls.xe_collision_set_v2(ion_neutral=True, **kwargs)

    def load_electron_cross_sections(self) -> XenonCrossSections:
        """Load and hash-check the declared electron set (what ``Simulation`` must be given)."""

        cross_sections = XenonCrossSections.from_file(spec_path(self.electron_file))
        self.check_electron(cross_sections)
        return cross_sections

    def check_electron(self, cross_sections: XenonCrossSections) -> None:
        if cross_sections.payload_sha256 != self.electron_payload_sha256:
            raise PIC2DValidationError(
                f"collision set {self.name!r} declares electron payload {self.electron_payload_sha256[:12]} but the supplied "
                f"cross sections have {cross_sections.payload_sha256[:12]}"
            )
        processes = tuple((p.identifier, p.kind, p.threshold_ev) for p in cross_sections.processes)
        if processes != self.electron_processes:
            raise PIC2DValidationError("collision set process list differs from the supplied cross sections")

    def load_ion_cross_sections(self) -> IonNeutralCrossSections | None:
        if self.ion_neutral is None:
            return None
        return self.ion_neutral.load()

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "electron_file": self.electron_file,
            "electron_payload_sha256": self.electron_payload_sha256,
            "electron_processes": [{"id": i, "kind": k, "threshold_ev": t} for i, k, t in self.electron_processes],
        }
        if self.ion_neutral is not None:
            record["ion_neutral"] = self.ion_neutral.to_dict()
        return record


def total_excitation_m2(cross_sections: XenonCrossSections, energy_ev: np.ndarray) -> np.ndarray:
    """``sum_k sigma_k(E)`` over the excitation levels (the lumped channel itself for the v1 set)."""

    e = np.asarray(energy_ev, dtype=np.float64)
    total = np.zeros_like(e)
    for process in cross_sections.excitation_levels:
        total = total + process.at(e)
    return total


def compare_excitation_sets(lumped: XenonCrossSections, resolved: XenonCrossSections,
                            energy_ev: np.ndarray | None = None) -> dict[str, Any]:
    """Total excitation cross section of the resolved set against the lumped one (the R3a attribution check).

    Returns the energies, both totals, the maximum relative deviation above 10 eV and the maximum absolute deviation
    (the per-level tables are anchored on their own thresholds, so the residual is grid interpolation across the sharp
    level onsets, not data).  Also the mean energy loss per excitation event of each set at a few electron energies,
    which is what actually changes between the sets.
    """

    if energy_ev is None:
        energy_ev = np.concatenate(([8.32, 9.0, 9.5, 10.0], np.geomspace(10.5, 1000.0, 200)))
    e = np.asarray(energy_ev, dtype=np.float64)
    sigma_lumped = total_excitation_m2(lumped, e)
    sigma_resolved = total_excitation_m2(resolved, e)
    mask = (e > 10.0) & (sigma_lumped > 0.0)
    relative = np.abs(sigma_resolved[mask] - sigma_lumped[mask]) / sigma_lumped[mask]
    mean_loss = {}
    for energy in (12.0, 20.0, 50.0, 100.0):
        weights = np.array([float(p.at(np.array([energy]))[0]) for p in resolved.excitation_levels])
        thresholds = np.array([p.threshold_ev for p in resolved.excitation_levels])
        mean_loss[f"{energy:g}"] = float(np.sum(weights * thresholds) / np.sum(weights)) if weights.sum() > 0 else float("nan")
    return {
        "energy_ev": e,
        "sigma_lumped_m2": sigma_lumped,
        "sigma_resolved_m2": sigma_resolved,
        "max_relative_deviation_above_10_ev": float(relative.max()) if relative.size else 0.0,
        "max_absolute_deviation_m2": float(np.max(np.abs(sigma_resolved - sigma_lumped))),
        "lumped_loss_ev": float(lumped.excitation_levels[0].threshold_ev),
        "resolved_mean_loss_ev": mean_loss,
    }


__all__ = [
    "XE_COLLISION_SET_V2_NAME",
    "XE_ELECTRON_SET_V2_FILE",
    "XE_ELECTRON_SET_V2_PAYLOAD_SHA256",
    "XE_ION_NEUTRAL_SET_V1_FILE",
    "XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256",
    "CollisionSetConfig",
    "compare_excitation_sets",
    "spec_path",
    "total_excitation_m2",
]
