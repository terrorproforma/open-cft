"""Model-to-model context from the pic2d steady-state v2 plateau (read-only).

The PIC run ``experiments/pic2d_cft_steady_state_v2`` (development,
single-seed, not validated) provides window-averaged maps of the divergent-exit
channel at 300 V with a 3.44 mA discharge current.  This module extracts, under
a DECLARED mapping, the quantities a four-cell model can be compared with:
plasma and dielectric-wall potentials at each magnetic cusp, the local electron
temperature, the electron and ion wall currents collected around each cusp, and
segment-averaged potentials/temperatures.  Nothing here validates anything: the
PIC is itself a development model and the comparison is context only.

Declared mapping (see ``docs/workstreams/plasma-v2-formulation.md`` section 6):
Kornfeld's cell 1 is the low-potential plume cell (his Table 3.1 remark),
cells 2-4 are the channel cells.  In the PIC channel the three on-axis
``B_z = 0`` planes sit at 17.95, 12.0 and 6.0 mm from the anode (README of the
run) and the divergent cone starts at 18 mm.  Hence: model cell 1 <-> cone
region [17.95, 24] mm; model cusp 2 (exit drop) <-> cusp at 17.95 mm; cell 2
<-> [12, 17.95] mm; cusp 3 <-> 12.0 mm; cell 3 <-> [6, 12] mm; the model's
anode-cusp electrons (p_4) <-> cusp at 6.0 mm; cell 4 <-> [0, 6] mm.  Model
cusp 1 (Kornfeld's plume-side cusp) has no magnetic counterpart; the dielectric
cone wall [18, 24] mm is reported beside it for completeness.

numpy is required only by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt
from pathlib import Path
from typing import Any

from .constants import ELEMENTARY_CHARGE_C

DEFAULT_MAPS_PATH = Path("experiments/pic2d_cft_steady_state_v2/results/maps.npz")
DEFAULT_SUMMARY_PATH = Path("experiments/pic2d_cft_steady_state_v2/results/summary.json")

# Geometry and cusp planes of the run (README / protocol of the experiment).
DR_M = 5.0e-5
DZ_M = 5.0e-5
BORE_RADIUS_M = 0.002
EXIT_RADIUS_M = 0.003
CONE_START_Z_M = 0.018
Z_MAX_M = 0.024
CUSP_PLANES_Z_M: tuple[float, float, float] = (0.01795, 0.012, 0.006)
SEGMENT_BOUNDS_Z_M: tuple[tuple[float, float], ...] = (
    (0.01795, 0.024),
    (0.012, 0.01795),
    (0.006, 0.012),
    (0.0, 0.006),
)
SEGMENT_LABELS: tuple[str, ...] = (
    "cell 1 (plume/cone) [17.95, 24] mm",
    "cell 2 [12, 17.95] mm",
    "cell 3 [6, 12] mm",
    "cell 4 (anode) [0, 6] mm",
)


@dataclass(frozen=True, slots=True)
class PicCuspContext:
    label: str
    z_m: float
    axis_potential_v: float
    near_wall_potential_v: float
    wall_potential_v: float
    sheath_drop_v: float
    near_wall_drop_v: float
    near_wall_electron_temperature_ev: float
    axis_electron_temperature_ev: float
    sheath_drop_over_near_wall_temperature: float
    electron_wall_current_a: float
    ion_wall_current_a: float
    electron_wall_mean_energy_ev: float
    window_half_width_m: float


@dataclass(frozen=True, slots=True)
class PicSegmentContext:
    label: str
    z_min_m: float
    z_max_m: float
    density_weighted_potential_v: float
    density_weighted_electron_temperature_ev: float
    mean_electron_density_per_m3: float
    peak_electron_density_per_m3: float


@dataclass(frozen=True, slots=True)
class PicPlateauContext:
    source_maps: str
    window_steps: int
    cusps: tuple[PicCuspContext, ...]
    cone_wall: PicCuspContext
    segments: tuple[PicSegmentContext, ...]
    potential_steps_v: tuple[float, ...]
    total_wall_electron_current_a: float
    total_wall_ion_current_a: float
    anode_potential_v: float
    phi_max_v: float
    phi_min_v: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_maps": self.source_maps,
            "window_steps": self.window_steps,
            "cusps": [asdict(cusp) for cusp in self.cusps],
            "cone_wall": asdict(self.cone_wall),
            "segments": [asdict(segment) for segment in self.segments],
            "potential_steps_v": list(self.potential_steps_v),
            "total_wall_electron_current_a": self.total_wall_electron_current_a,
            "total_wall_ion_current_a": self.total_wall_ion_current_a,
            "anode_potential_v": self.anode_potential_v,
            "phi_max_v": self.phi_max_v,
            "phi_min_v": self.phi_min_v,
        }


def wall_radius_m(z_m: float) -> float:
    if z_m <= CONE_START_Z_M:
        return BORE_RADIUS_M
    return BORE_RADIUS_M + (EXIT_RADIUS_M - BORE_RADIUS_M) * (z_m - CONE_START_Z_M) / (
        Z_MAX_M - CONE_START_Z_M
    )


def _slant_factor(z_m: float) -> float:
    if z_m <= CONE_START_Z_M:
        return 1.0
    slope = (EXIT_RADIUS_M - BORE_RADIUS_M) / (Z_MAX_M - CONE_START_Z_M)
    return sqrt(1.0 + slope * slope)


def load_pic_plateau_context(
    maps_path: Path,
    *,
    anode_potential_v: float = 300.0,
    window_half_width_m: float = 1.0e-3,
    near_wall_band_m: float = 5.0e-4,
) -> PicPlateauContext:
    """Extract the comparison quantities from the window-averaged maps."""

    import numpy as np

    maps = np.load(maps_path)
    phi = np.asarray(maps["phi_v"], dtype=float)
    t_e = np.asarray(maps["t_e_ev"], dtype=float)
    n_e = np.asarray(maps["n_e_per_m3"], dtype=float)
    wall_e = np.asarray(maps["wall_electron_flux_per_m2_s"], dtype=float)
    wall_i = np.asarray(maps["wall_ion_flux_per_m2_s"], dtype=float)
    wall_e_energy = np.asarray(maps["wall_electron_mean_energy_ev"], dtype=float)
    window_steps = int(np.asarray(maps["window_steps"]).ravel()[0])
    radial_nodes, axial_nodes = phi.shape
    z_nodes = np.arange(axial_nodes) * DZ_M
    r_nodes = np.arange(radial_nodes) * DR_M
    z_cells = (np.arange(wall_e.size) + 0.5) * DZ_M
    area = np.array([2.0 * pi * wall_radius_m(z) * DZ_M * _slant_factor(z) for z in z_cells])
    current_e = ELEMENTARY_CHARGE_C * wall_e * area
    current_i = ELEMENTARY_CHARGE_C * wall_i * area

    def cusp_context(label: str, z_c: float, half_width: float) -> PicCuspContext:
        j = int(round(z_c / DZ_M))
        wall_index = int(round(wall_radius_m(z_c) / DR_M))
        wall_index = min(wall_index, radial_nodes - 1)
        axis_phi = float(phi[0, j])
        wall_phi = float(phi[wall_index, j])
        # Under-resolved proxy for the sheath-edge potential: three cells
        # (150 um, ~9 Debye lengths at the peak density) inside the wall.
        near_wall_phi = float(phi[max(0, wall_index - 3), j])
        band = max(1, int(round(near_wall_band_m / DR_M)))
        j_lo = max(0, int(round((z_c - half_width) / DZ_M)))
        j_hi = min(axial_nodes, int(round((z_c + half_width) / DZ_M)) + 1)
        r_lo = max(0, wall_index - band)
        block_t = t_e[r_lo:wall_index, j_lo:j_hi]
        block_n = n_e[r_lo:wall_index, j_lo:j_hi]
        weight = block_n.sum()
        near_wall_t = float((block_t * block_n).sum() / weight) if weight > 0 else float("nan")
        axis_t = float(t_e[0, j])
        mask = (z_cells >= z_c - half_width) & (z_cells <= z_c + half_width)
        i_e = float(current_e[mask].sum())
        i_i = float(current_i[mask].sum())
        energy_weight = (wall_e * area)[mask].sum()
        mean_energy = (
            float(((wall_e * area * wall_e_energy)[mask]).sum() / energy_weight)
            if energy_weight > 0
            else float("nan")
        )
        drop = axis_phi - wall_phi
        return PicCuspContext(
            label=label,
            z_m=z_c,
            axis_potential_v=axis_phi,
            near_wall_potential_v=near_wall_phi,
            wall_potential_v=wall_phi,
            sheath_drop_v=drop,
            near_wall_drop_v=near_wall_phi - wall_phi,
            near_wall_electron_temperature_ev=near_wall_t,
            axis_electron_temperature_ev=axis_t,
            sheath_drop_over_near_wall_temperature=(drop / near_wall_t if near_wall_t > 0 else float("nan")),
            electron_wall_current_a=i_e,
            ion_wall_current_a=i_i,
            electron_wall_mean_energy_ev=mean_energy,
            window_half_width_m=half_width,
        )

    cusps = tuple(
        cusp_context(f"PIC cusp {index + 1} (z = {z * 1e3:.2f} mm)", z, window_half_width_m)
        for index, z in enumerate(CUSP_PLANES_Z_M)
    )
    cone_centre = 0.5 * (CONE_START_Z_M + Z_MAX_M)
    cone_wall = cusp_context("PIC dielectric cone wall [18, 24] mm", cone_centre, 0.5 * (Z_MAX_M - CONE_START_Z_M))

    segments: list[PicSegmentContext] = []
    for label, (z_lo, z_hi) in zip(SEGMENT_LABELS, SEGMENT_BOUNDS_Z_M, strict=True):
        cols = (z_nodes >= z_lo) & (z_nodes <= z_hi)
        block_n = n_e[:, cols]
        block_phi = phi[:, cols]
        block_t = t_e[:, cols]
        # cylindrical volume weight r dr dz (axis node weight -> dr/8 via r = dr/4 surrogate)
        radial_weight = np.where(r_nodes > 0, r_nodes, 0.25 * DR_M)[:, None]
        weight = block_n * radial_weight
        total = weight.sum()
        segments.append(
            PicSegmentContext(
                label=label,
                z_min_m=z_lo,
                z_max_m=z_hi,
                density_weighted_potential_v=float((block_phi * weight).sum() / total) if total > 0 else float("nan"),
                density_weighted_electron_temperature_ev=float((block_t * weight).sum() / total) if total > 0 else float("nan"),
                mean_electron_density_per_m3=float((block_n * radial_weight).sum() / (radial_weight.sum() * block_n.shape[1])),
                peak_electron_density_per_m3=float(block_n.max()),
            )
        )
    steps = tuple(
        segments[index + 1].density_weighted_potential_v - segments[index].density_weighted_potential_v
        for index in range(len(segments) - 1)
    )
    return PicPlateauContext(
        source_maps=maps_path.as_posix(),
        window_steps=window_steps,
        cusps=cusps,
        cone_wall=cone_wall,
        segments=tuple(segments),
        potential_steps_v=steps,
        total_wall_electron_current_a=float(current_e.sum()),
        total_wall_ion_current_a=float(current_i.sum()),
        anode_potential_v=anode_potential_v,
        phi_max_v=float(phi.max()),
        phi_min_v=float(phi.min()),
    )
