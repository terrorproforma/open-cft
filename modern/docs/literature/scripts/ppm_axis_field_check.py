"""Check recorded CFT field maps against the analytic PPM (TWT) axis-field theory.

Read-only on every artifact. For the four L1a geometry-sweep-v2 representatives
(000/032/065/068) and, if loadable, the P2 divergent-exit FEM field, the script

1. fits the classic periodic-permanent-magnet axis form
       B_z(0, z) = sum_{k odd} b_k cos(k*pi*(z - z0)/L + phi)
   to the recorded axis B_z over the stack interior (first to last stage centre),
   with L (half-period = stage pitch), phi and the b_k free;
2. reports the fitted pitch against the geometric pitch, the harmonic content
   b_3/b_1 and b_5/b_1 and the RMS misfit;
3. predicts the cusp (axis-null) positions from the fitted series and compares
   them with the recorded axis nulls, with the stage gaps (pole-piece planes)
   and, when a topology-v3 checkout is given, with the v3 separatrix wall
   intersections z_c;
4. extends the fitted series off axis with the exact Laplace solution
       B_z = sum b_k I_0(k kappa r) cos(.),  B_r = sum b_k I_1(k kappa r) sin(.),
   kappa = pi/L, and compares the paraxial (-r/2 dB_z/dz) and Bessel wall B_r
   with the recorded wall |B_r| maxima;
5. reports the implied mirror ratios (wall cusp field over axis peak, over the
   wall field at the stage centre, and along the launch field lines of the
   orbit campaigns), the electron adiabaticity numbers (Larmor radius,
   cyclotron wavelength per period, Mendel parameter, non-adiabatic radius);
6. locates, from the geometry-screening endpoint tables, where the recorded
   reflections happened relative to the stage centres and the axis nulls.

Usage (from ``modern/``, CPU only)::

    $env:PYTHONPATH="$PWD\\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
    python docs/literature/scripts/ppm_axis_field_check.py [--topology-v3-root PATH]
        [--json OUT.json] [--no-p2]

Nothing under ``experiments/`` or ``examples/`` is written.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

MODERN = Path(__file__).resolve().parents[3]
REPOSITORY = MODERN.parent
SWEEP = MODERN / "experiments" / "l1a_geometry_sweep_v2" / "results" / "representatives"
SCREENING = MODERN / "experiments" / "orbit_wall_loss_geometry_screening_v1" / "results" / "artifacts"
V4_PROTOCOL = MODERN / "experiments" / "cft_orbit_wall_loss_v4" / "protocol.json"
REPRESENTATIVES = (
    "l1a-gs-v2-000-48d2ccedd5",
    "l1a-gs-v2-032-570ad83ba6",
    "l1a-gs-v2-065-9e98f08f3b",
    "l1a-gs-v2-068-375d1b1b13",
)
HARMONICS = (1, 3, 5)
ELECTRON_MASS = 9.1093837015e-31
ELEMENTARY_CHARGE = 1.602176634e-19
LAUNCH_ENERGIES_EV = (5.0, 25.0)
LAUNCH_RADIUS_FRACTIONS = (0.675, 0.8)


# ---- Bessel functions without scipy ------------------------------------------


def bessel_i(order: int, x: np.ndarray | float) -> np.ndarray:
    """Modified Bessel function I_order(x) by its power series (x <= ~30)."""
    x = np.asarray(x, dtype=np.float64)
    half = 0.5 * x
    term = half**order / math.factorial(order)
    total = np.array(term, dtype=np.float64, copy=True)
    for m in range(1, 80):
        term = term * half * half / (m * (m + order))
        total = total + term
        if np.all(np.abs(term) <= 1e-17 * np.abs(total) + 1e-300):
            break
    return total


# ---- PPM series model ---------------------------------------------------------


class PPMSeries:
    """B_z(0,z) = sum_k b_k cos(k kappa (z - z0) + phi), kappa = pi / L."""

    def __init__(self, pitch_m: float, z0_m: float, phi: float, b: dict[int, float]):
        self.pitch_m = float(pitch_m)
        self.z0_m = float(z0_m)
        self.phi = float(phi)
        self.b = {int(k): float(v) for k, v in b.items()}
        self.kappa = math.pi / self.pitch_m

    def _arg(self, k: int, z):
        return k * self.kappa * (np.asarray(z, dtype=np.float64) - self.z0_m) + self.phi

    def bz(self, r, z):
        r = np.asarray(r, dtype=np.float64)
        return sum(bk * bessel_i(0, k * self.kappa * r) * np.cos(self._arg(k, z)) for k, bk in self.b.items())

    def br(self, r, z):
        r = np.asarray(r, dtype=np.float64)
        return sum(bk * bessel_i(1, k * self.kappa * r) * np.sin(self._arg(k, z)) for k, bk in self.b.items())

    def br_paraxial(self, r, z):
        """-(r/2) dB_z(0,z)/dz."""
        return -0.5 * np.asarray(r, dtype=np.float64) * self.dbz_dz_axis(z)

    def dbz_dz_axis(self, z):
        return sum(-bk * k * self.kappa * np.sin(self._arg(k, z)) for k, bk in self.b.items())

    def dbz_dz(self, r, z):
        r = np.asarray(r, dtype=np.float64)
        return sum(-bk * k * self.kappa * bessel_i(0, k * self.kappa * r) * np.sin(self._arg(k, z)) for k, bk in self.b.items())

    def psi(self, r, z):
        """Flux function r*A_phi with B_z = (1/r) dpsi/dr, B_r = -(1/r) dpsi/dz."""
        r = np.asarray(r, dtype=np.float64)
        return sum(bk * r / (k * self.kappa) * bessel_i(1, k * self.kappa * r) * np.cos(self._arg(k, z)) for k, bk in self.b.items())

    def magnitude(self, r, z):
        return np.hypot(self.br(r, z), self.bz(r, z))

    def axis_roots(self, z_lo: float, z_hi: float, samples: int = 4001) -> list[float]:
        z = np.linspace(z_lo, z_hi, samples)
        f = self.bz(0.0, z)
        roots = []
        for i in range(samples - 1):
            if f[i] == 0.0:
                roots.append(float(z[i]))
            elif f[i] * f[i + 1] < 0.0:
                a, fa, c = float(z[i]), float(f[i]), float(z[i + 1])
                for _ in range(60):
                    mid = 0.5 * (a + c)
                    fm = float(self.bz(0.0, mid))
                    if fa * fm <= 0.0:
                        c = mid
                    else:
                        a, fa = mid, fm
                roots.append(0.5 * (a + c))
        return roots


def fit_ppm_series(z: np.ndarray, bz: np.ndarray, pitch_guess: float, z0: float, harmonics=HARMONICS) -> PPMSeries:
    """Least-squares fit of the odd-harmonic cosine series with free pitch and phase.

    Outer grid+refine over (L, phi); inner linear least squares for the b_k.
    """

    def design(pitch: float, phi: float) -> np.ndarray:
        kappa = math.pi / pitch
        return np.column_stack([np.cos(k * kappa * (z - z0) + phi) for k in harmonics])

    def residual(pitch: float, phi: float) -> tuple[float, np.ndarray]:
        matrix = design(pitch, phi)
        coeffs, *_ = np.linalg.lstsq(matrix, bz, rcond=None)
        return float(np.sqrt(np.mean((matrix @ coeffs - bz) ** 2))), coeffs

    best = (math.inf, pitch_guess, 0.0, None)
    for pitch in np.linspace(0.8 * pitch_guess, 1.25 * pitch_guess, 91):
        for phi in np.linspace(-math.pi / 2, math.pi / 2, 73):
            rms, coeffs = residual(float(pitch), float(phi))
            if rms < best[0]:
                best = (rms, float(pitch), float(phi), coeffs)
    rms, pitch, phi, coeffs = best
    step_p, step_f = 0.005 * pitch_guess, math.pi / 72
    for _ in range(40):
        improved = False
        for dp, df in ((step_p, 0.0), (-step_p, 0.0), (0.0, step_f), (0.0, -step_f)):
            trial_rms, trial_coeffs = residual(pitch + dp, phi + df)
            if trial_rms < rms:
                rms, pitch, phi, coeffs = trial_rms, pitch + dp, phi + df, trial_coeffs
                improved = True
        if not improved:
            step_p *= 0.5
            step_f *= 0.5
    series = PPMSeries(pitch, z0, phi, dict(zip(harmonics, coeffs)))
    series.rms_t = rms  # type: ignore[attr-defined]
    return series


# ---- loaders ----------------------------------------------------------------


def load_sweep_representative(case_id: str) -> dict:
    field = json.loads(next(SWEEP.glob(f"{case_id}.field-downsampled.json")).read_text(encoding="utf-8"))
    geometry = json.loads(next(SWEEP.glob(f"{case_id}.geometry.json")).read_text(encoding="utf-8"))
    stages = geometry["stages"]
    magnet = next(r for r in geometry["regions"] if r["role"] == "permanent_magnet")
    poles = [r for r in geometry["regions"] if r["role"] == "pole_piece"]
    full_map = None
    full_path = SWEEP / f"{case_id}.field-full.json"
    if full_path.exists():
        fm = json.loads(full_path.read_text(encoding="utf-8"))["field_map"]
        full_map = {
            "r_m": np.asarray(fm["r_m"], dtype=np.float64),
            "z_m": np.asarray(fm["z_m"], dtype=np.float64),
            "b_t": np.asarray(fm["b_magnitude_t"], dtype=np.float64),
        }
    return {
        "design_id": case_id,
        "level": "L1a linear-vacuum equivalent-current FDM (no iron in the field)",
        "full_map": full_map,
        "axis_z_m": np.asarray(field["profiles"]["centreline"]["z_m"], dtype=np.float64),
        "axis_bz_t": np.asarray(field["profiles"]["centreline"]["b_z_t"], dtype=np.float64),
        "wall_z_m": np.asarray(field["profiles"]["wall"]["z_m"], dtype=np.float64),
        "wall_br_t": np.asarray(field["profiles"]["wall"]["b_r_t"], dtype=np.float64),
        "wall_bz_t": np.asarray(field["profiles"]["wall"]["b_z_t"], dtype=np.float64),
        "wall_sampled_r_m": float(field["profiles"]["wall"]["sampled_r_m"]),
        "wall_radius_m": float(geometry["chamber"]["outer_radius_m"]),
        "chamber_length_m": float(geometry["chamber"]["length_m"]),
        "stage_pitch_m": float(stages[0]["pitch_m"]),
        "stage_centres_m": [float(s["center_z_m"]) for s in stages],
        "magnet_inner_radius_m": float(magnet["r_inner_start_m"]),
        "magnet_axial_thickness_m": float(magnet["z_max_m"] - magnet["z_min_m"]),
        "pole_gap_m": float(poles[0]["z_max_m"] - poles[0]["z_min_m"]) if poles else float("nan"),
        "recorded_axis_nulls_m": [float(n["z_m"]) for n in field["summary"]["topology"]["axis_nulls"]],
        "axis_dz_m": float(field["input"]["domain"]["dz_m"]),
    }


def load_p2() -> dict | None:
    try:
        sys.path.insert(0, str(MODERN))
        sys.path.insert(0, str(MODERN / "src"))
        from experiments.cft_orbit_wall_loss_v4 import adapter as v4_adapter  # noqa: PLC0415
        from cft_revival.geometry.generators import divergent_exit_stack  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"P2 adapter not importable ({exc!r}); skipping P2", file=sys.stderr)
        return None
    protocol = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    fa = protocol["field_adapter"]
    declaration = fa["maps"]["primary"]
    checkpoint = REPOSITORY / declaration["checkpoint_path"]
    if not checkpoint.exists():
        print("P2 checkpoint absent; skipping P2", file=sys.stderr)
        return None
    try:
        evaluator = v4_adapter.BoundP2Evaluator(
            checkpoint, declaration, allowed_regions=set(fa["plasma_region_ids"]), bounds=fa["regular_plasma_domain"]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"P2 checkpoint not loadable ({exc!r}); skipping P2", file=sys.stderr)
        return None
    design = divergent_exit_stack()
    bounds = fa["regular_plasma_domain"]
    z = np.linspace(bounds["z_min_m"], bounds["z_max_m"], 441)
    wall_r = float(design.chamber.outer_radius_m)
    axis = np.array([evaluator.evaluate(0.0, float(zz)) for zz in z])
    wall = np.array([evaluator.evaluate(wall_r, float(zz)) for zz in z])
    magnet = next(r for r in design.regions if str(getattr(r.role, "value", r.role)) == "permanent_magnet")
    stages = design.stages
    return {
        "design_id": "divergent-exit-stack (P2, level-1 checkpoint)",
        "level": "P2 adaptive FEM, iron pole rings and return yoke in the source model",
        "axis_z_m": z,
        "axis_bz_t": axis[:, 2],
        "wall_z_m": z,
        "wall_br_t": wall[:, 1],
        "wall_bz_t": wall[:, 2],
        "wall_sampled_r_m": wall_r,
        "wall_radius_m": wall_r,
        "chamber_length_m": float(design.chamber.length_m),
        "stage_pitch_m": float(stages[0].pitch_m),
        "stage_centres_m": [float(s.center_z_m) for s in stages],
        "magnet_inner_radius_m": float(magnet.r_inner_start_m),
        "magnet_axial_thickness_m": float(magnet.z_max_m - magnet.z_min_m),
        "pole_gap_m": float(stages[0].pitch_m - (magnet.z_max_m - magnet.z_min_m)),
        "recorded_axis_nulls_m": [],  # derived below from the sampled axis
        "axis_dz_m": float(z[1] - z[0]),
        "straight_wall_z_max_m": float(protocol["orbit"]["wall"]["z_max_m"]),
    }


def sampled_axis_nulls(z: np.ndarray, bz: np.ndarray) -> list[float]:
    roots = []
    for i in range(len(z) - 1):
        if bz[i] == 0.0:
            roots.append(float(z[i]))
        elif bz[i] * bz[i + 1] < 0.0:
            roots.append(float(z[i] - bz[i] * (z[i + 1] - z[i]) / (bz[i + 1] - bz[i])))
    return roots


TOPOLOGY_EXPERIMENTS = ("cusp_topology_search_v3_1", "cusp_topology_search_v3")


def load_topology_v3(root: Path | None) -> dict:
    """Load the cusp/cell catalogue of the topology search (v3.1 preferred, v3 as lineage).

    ``root`` is a checkout; by default this repository. The first experiment directory
    that holds a catalogue is used and its recorded terminal state is reported.
    """
    root = REPOSITORY if root is None else root
    for experiment in TOPOLOGY_EXPERIMENTS:
        artifacts = root / "modern" / "experiments" / experiment / "results" / "artifacts"
        catalogue = artifacts / "cusp-cell-catalogue.json"
        if not catalogue.exists():
            continue
        data = json.loads(catalogue.read_text(encoding="utf-8"))
        state = None
        result = artifacts / "campaign-result.json"
        if result.exists():
            payload = json.loads(result.read_text(encoding="utf-8"))
            state = payload.get("terminal_state") or payload.get("state") or payload.get("status")
        return {
            "experiment": experiment,
            "experiment_id": data.get("experiment_id"),
            "state": state,
            "entries": {e["design_id"]: e for e in data["entries"]},
        }
    print(f"no topology catalogue found under {root}", file=sys.stderr)
    return {}


def load_reflections(case_id: str) -> dict | None:
    path = SCREENING / "endpoints" / f"{case_id}--accepted-2N.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    reflected = [r for r in rows if r["termination"] == "reflected"]
    per_cell = {}
    for r in rows:
        cell = per_cell.setdefault(r["cell_id"], {"launch_z_m": float(r["launch_z_m"]), "orbits": 0, "reflected": 0, "wall_hit": 0})
        cell["orbits"] += 1
        cell[r["termination"]] = cell.get(r["termination"], 0) + 1
    return {
        "per_cell": per_cell,
        "reflected_z_m": np.array([r["final_z_m"] for r in reflected]),
        "reflected_r_m": np.array([r["final_r_m"] for r in reflected]),
        "reflected_launch_z_m": np.array([r["launch_z_m"] for r in reflected]),
        "reflected_launch_r_m": np.array([r["launch_r_m"] for r in reflected]),
        "wall_hit_z_m": np.array([r["final_z_m"] for r in rows if r["termination"] == "wall_hit"]),
        "mu_median": float(np.median([r["mu_relative_variation"] for r in rows])),
        "count": len(rows),
        "reflected_pitch_deg": np.array([r["pitch_angle_deg"] for r in reflected]),
    }


def interpolate_map(full_map: dict, r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of |B| on the recorded structured map."""
    rs, zs, values = full_map["r_m"], full_map["z_m"], full_map["b_t"]
    i = np.clip(np.searchsorted(rs, r) - 1, 0, len(rs) - 2)
    j = np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2)
    tr = (r - rs[i]) / (rs[i + 1] - rs[i])
    tz = (z - zs[j]) / (zs[j + 1] - zs[j])
    return (
        values[i, j] * (1 - tr) * (1 - tz)
        + values[i + 1, j] * tr * (1 - tz)
        + values[i, j + 1] * (1 - tr) * tz
        + values[i + 1, j + 1] * tr * tz
    )


# ---- analysis ---------------------------------------------------------------


def nearest_distance(values: np.ndarray, anchors: list[float]) -> np.ndarray:
    anchors_arr = np.asarray(anchors, dtype=np.float64)
    return np.min(np.abs(values[:, None] - anchors_arr[None, :]), axis=1)


def analyse(design: dict, v3_entry: dict | None) -> dict:
    z, bz = design["axis_z_m"], design["axis_bz_t"]
    centres = design["stage_centres_m"]
    pitch = design["stage_pitch_m"]
    window = (centres[0], centres[-1])
    mask = (z >= window[0] - 1e-12) & (z <= window[1] + 1e-12)
    z0 = 0.5 * (centres[0] + centres[-1])
    series = fit_ppm_series(z[mask], bz[mask], pitch, z0)
    b1 = series.b[1]
    axis_peak = float(np.max(np.abs(bz[mask])))

    # cusps
    gaps = [0.5 * (a + b) for a, b in zip(centres[:-1], centres[1:])]
    predicted = series.axis_roots(window[0], window[1])
    recorded = design["recorded_axis_nulls_m"] or sampled_axis_nulls(z, bz)
    recorded_interior = [n for n in recorded if window[0] < n < window[1]]
    matched = []
    for gap in gaps:
        rec = min(recorded_interior, key=lambda n: abs(n - gap)) if recorded_interior else float("nan")
        pred = min(predicted, key=lambda n: abs(n - gap)) if predicted else float("nan")
        v3 = float("nan")
        v3_angle = float("nan")
        v3_wall_b = float("nan")
        if v3_entry:
            cusps = v3_entry["wall_cusps"]
            if cusps:
                best = min(cusps, key=lambda c: abs(c["z_c_m"] - gap))
                v3, v3_angle, v3_wall_b = float(best["z_c_m"]), float(best["angle_to_wall_normal_deg"]), float(best["wall_b_t"])
        matched.append({"gap_m": gap, "recorded_null_m": rec, "fitted_null_m": pred, "v3_z_c_m": v3, "v3_angle_deg": v3_angle, "v3_wall_b_t": v3_wall_b})

    # wall B_r
    r_w = design["wall_sampled_r_m"]
    wz, wbr = design["wall_z_m"], design["wall_br_t"]
    wmask = (wz >= window[0] - 1e-12) & (wz <= window[1] + 1e-12)
    recorded_wall_br_max = float(np.max(np.abs(wbr[wmask])))
    zfine = np.linspace(window[0], window[1], 4001)
    paraxial_max = float(np.max(np.abs(series.br_paraxial(r_w, zfine))))
    bessel_max = float(np.max(np.abs(series.br(r_w, zfine))))
    fundamental_only = PPMSeries(series.pitch_m, series.z0_m, series.phi, {1: series.b[1]})
    fundamental_wall_max = float(np.max(np.abs(fundamental_only.br(r_w, zfine))))
    x_w = math.pi * r_w / series.pitch_m
    # harmonic content actually present at the wall: fit the recorded wall B_r with the
    # Bessel-extended series at the axis-fitted (L, z0, phi), coefficients free
    wall_matrix = np.column_stack([
        bessel_i(1, k * series.kappa * r_w) * np.sin(k * series.kappa * (wz[wmask] - series.z0_m) + series.phi) for k in HARMONICS
    ])
    wall_coeffs, *_ = np.linalg.lstsq(wall_matrix, wbr[wmask], rcond=None)
    wall_fit_rms = float(np.sqrt(np.mean((wall_matrix @ wall_coeffs - wbr[wmask]) ** 2)))
    wall_series = PPMSeries(series.pitch_m, series.z0_m, series.phi, dict(zip(HARMONICS, wall_coeffs)))
    wall_harmonics = {
        f"wall_b{k}_over_wall_b1": float(wall_coeffs[i] / wall_coeffs[0]) for i, k in enumerate(HARMONICS)
    }
    wall_harmonics["wall_b1_over_axis_b1"] = float(wall_coeffs[0] / b1)
    wall_harmonics["wall_fit_rms_over_max"] = wall_fit_rms / recorded_wall_br_max
    # sharpness of the wall cusp: |B_r| full width at half maximum around the first interior cusp,
    # recorded vs fundamental-only prediction (FWHM of |sin| = 2L/3 for a pure fundamental)
    wall_fwhm_recorded = float("nan")
    if len(gaps):
        zc0 = min(recorded_interior, key=lambda n: abs(n - gaps[0])) if recorded_interior else gaps[0]
        zz = np.linspace(zc0 - 0.5 * pitch, zc0 + 0.5 * pitch, 2001)
        prof = np.abs(np.interp(zz, wz, wbr))
        half = 0.5 * float(np.max(prof))
        above = zz[prof >= half]
        if len(above) > 1:
            wall_fwhm_recorded = float(above[-1] - above[0])
    wall_fwhm_fundamental = 2.0 * pitch / 3.0

    # wall-field angle and mirror ratios at the interior cusps
    cusp_rows = []
    for m in matched:
        zc = m["recorded_null_m"]
        if math.isnan(zc):
            continue
        rec_br = float(np.interp(zc, wz, wbr))
        rec_bz = float(np.interp(zc, wz, design["wall_bz_t"]))
        angle_rec = math.degrees(math.atan2(abs(rec_bz), abs(rec_br)))
        fit_br, fit_bz = float(series.br(r_w, zc)), float(series.bz(r_w, zc))
        angle_fit = math.degrees(math.atan2(abs(fit_bz), abs(fit_br)))
        # neighbouring axis peaks
        left = max([c for c in centres if c < zc], default=None)
        right = min([c for c in centres if c > zc], default=None)
        peaks = []
        for c in (left, right):
            if c is not None:
                i = int(np.argmin(np.abs(z - c)))
                peaks.append(abs(float(bz[i])))
        axis_peak_adjacent = max(peaks) if peaks else float("nan")
        wall_mag_cusp = math.hypot(rec_br, rec_bz)
        jw = int(np.argmin(np.abs(wz - (left if left is not None else right))))
        wall_mag_centre = math.hypot(float(wbr[jw]), float(design["wall_bz_t"][jw]))
        cusp_rows.append({
            "z_c_m": zc,
            "recorded_wall_br_t": rec_br,
            "recorded_wall_b_t": wall_mag_cusp,
            "fitted_wall_br_t": fit_br,
            "angle_recorded_deg": angle_rec,
            "angle_fitted_deg": angle_fit,
            "ratio_wall_cusp_over_axis_peak": wall_mag_cusp / axis_peak_adjacent,
            "ratio_wall_cusp_over_wall_centre": wall_mag_cusp / wall_mag_centre,
        })

    # field-line mirror check along the launch field lines (fitted series)
    line_rows = []
    zc_list = [m["recorded_null_m"] for m in matched if not math.isnan(m["recorded_null_m"])]
    if len(centres) >= 3 and zc_list:
        c_mid = centres[len(centres) // 2]
        zc_next = min([c for c in zc_list if c > c_mid], default=None)
        if zc_next is not None:
            for frac in LAUNCH_RADIUS_FRACTIONS:
                r0 = frac * design["wall_radius_m"]
                psi0 = float(series.psi(r0, c_mid))
                b0 = float(series.magnitude(r0, c_mid))
                zs = np.linspace(c_mid, zc_next, 2001)
                psi_wall = series.psi(r_w, zs) - psi0
                hit = None
                for i in range(len(zs) - 1):
                    if psi_wall[i] * psi_wall[i + 1] <= 0.0:
                        hit = float(zs[i] - psi_wall[i] * (zs[i + 1] - zs[i]) / (psi_wall[i + 1] - psi_wall[i]))
                        break
                if hit is None:
                    line_rows.append({"launch_r_over_rw": frac, "reaches_wall": False})
                    continue
                b_wall = float(series.magnitude(r_w, hit))
                # maximum |B| along the traced line between launch and wall hit; psi is
                # monotonic in r with the sign of B_z at the launch plane
                sign = 1.0 if float(series.bz(0.0, c_mid)) >= 0.0 else -1.0
                zline = np.linspace(c_mid, hit, 400)
                rline = []
                for zz in zline:
                    lo, hi = 0.0, r_w
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        if sign * float(series.psi(mid, zz)) < sign * psi0:
                            lo = mid
                        else:
                            hi = mid
                    rline.append(0.5 * (lo + hi))
                bmax_line = float(np.max(series.magnitude(np.asarray(rline), zline)))
                line_rows.append({
                    "launch_r_over_rw": frac,
                    "reaches_wall": True,
                    "wall_hit_fraction_to_cusp": (hit - c_mid) / (zc_next - c_mid),
                    "b_launch_t": b0,
                    "b_wall_t": b_wall,
                    "ratio_wall_over_launch": b_wall / b0,
                    "max_along_line_over_launch": bmax_line / b0,
                })

    # electron numbers
    electron_rows = []
    b_cusp_wall = float(np.mean([c["recorded_wall_b_t"] for c in cusp_rows])) if cusp_rows else float("nan")
    for energy in LAUNCH_ENERGIES_EV:
        v = math.sqrt(2.0 * energy * ELEMENTARY_CHARGE / ELECTRON_MASS)
        r_l_peak = ELECTRON_MASS * v / (ELEMENTARY_CHARGE * abs(b1))
        r_l_cusp_wall = ELECTRON_MASS * v / (ELEMENTARY_CHARGE * b_cusp_wall) if b_cusp_wall else float("nan")
        lambda_c = 2.0 * math.pi * r_l_peak
        period = 2.0 * series.pitch_m
        alpha = (period / lambda_c) ** 2 / 8.0
        d_na = math.sqrt(ELECTRON_MASS * v / (ELEMENTARY_CHARGE * abs(b1) * series.kappa))
        # local scale length at the wall cusp: |B| / |dB_z/dz|
        if zc_list:
            zc = zc_list[0]
            scale = float(series.magnitude(r_w, zc) / max(abs(series.dbz_dz(r_w, zc)), 1e-300))
        else:
            scale = float("nan")
        electron_rows.append({
            "energy_ev": energy,
            "r_l_at_axis_peak_m": r_l_peak,
            "r_l_at_wall_cusp_m": r_l_cusp_wall,
            "period_over_lambda_c": period / lambda_c,
            "mendel_alpha": alpha,
            "nonadiabatic_radius_m": d_na,
            "wall_cusp_scale_length_m": scale,
            "epsilon_wall_cusp": r_l_cusp_wall / scale if scale else float("nan"),
        })

    return {
        "design_id": design["design_id"],
        "level": design["level"],
        "stage_count": len(centres),
        "geometric_pitch_m": pitch,
        "fitted_pitch_m": series.pitch_m,
        "pitch_ratio": series.pitch_m / pitch,
        "fitted_phase_rad": series.phi,
        "b1_t": b1,
        "b3_over_b1": series.b[3] / b1,
        "b5_over_b1": series.b[5] / b1,
        "fit_rms_t": series.rms_t,  # type: ignore[attr-defined]
        "fit_rms_over_peak": series.rms_t / axis_peak,  # type: ignore[attr-defined]
        "axis_dz_m": design["axis_dz_m"],
        "points_per_pitch": pitch / design["axis_dz_m"],
        "axis_peak_t": axis_peak,
        "wall_sampled_r_m": r_w,
        "wall_radius_m": design["wall_radius_m"],
        "x_w": x_w,
        "I1_xw": float(bessel_i(1, x_w)),
        "I1_over_I0_xw": float(bessel_i(1, x_w) / bessel_i(0, x_w)),
        "magnet_inner_radius_over_pitch": design["magnet_inner_radius_m"] / pitch,
        "magnet_axial_fraction": design["magnet_axial_thickness_m"] / pitch,
        "recorded_wall_br_max_t": recorded_wall_br_max,
        "paraxial_wall_br_max_t": paraxial_max,
        "bessel_wall_br_max_t": bessel_max,
        "fundamental_wall_br_max_t": fundamental_wall_max,
        "wall_harmonics": wall_harmonics,
        "wall_fwhm_recorded_m": wall_fwhm_recorded,
        "wall_fwhm_fundamental_m": wall_fwhm_fundamental,
        "cusps": matched,
        "cusp_fields": cusp_rows,
        "field_lines": line_rows,
        "electrons": electron_rows,
        "v3_state": None,
        "v3_axis_mirror_ratios": [c["axis_mirror_ratio"] for c in v3_entry["cells"]] if v3_entry else None,
        "v3_wall_mirror_ratios": [c["wall_mirror_ratio"] for c in v3_entry["cells"]] if v3_entry else None,
    }


def reflection_summary(design: dict, analysis: dict, refl: dict | None) -> dict | None:
    if refl is None:
        return None
    centres = design["stage_centres_m"]
    nulls = design["recorded_axis_nulls_m"]
    pitch = design["stage_pitch_m"]
    zr = refl["reflected_z_m"]
    zw = refl["wall_hit_z_m"]
    out = {"orbits": refl["count"], "reflected": int(len(zr)), "wall_hits": int(len(zw)), "mu_median": refl["mu_median"]}
    cells = []
    for cell_id in sorted(refl["per_cell"]):
        cell = refl["per_cell"][cell_id]
        zl = cell["launch_z_m"]
        d_null = float(np.min(np.abs(np.asarray(nulls) - zl))) / pitch if nulls else float("nan")
        d_centre = float(np.min(np.abs(np.asarray(centres) - zl))) / pitch
        b_launch = float(np.interp(zl, design["axis_z_m"], np.abs(design["axis_bz_t"])))
        cells.append({
            "cell_id": cell_id,
            "launch_z_m": zl,
            "dist_to_null_over_pitch": d_null,
            "dist_to_centre_over_pitch": d_centre,
            "axis_b_at_launch_over_peak": b_launch / analysis["axis_peak_t"],
            "orbits": cell["orbits"],
            "reflected": cell.get("reflected", 0),
            "wall_hit": cell.get("wall_hit", 0),
        })
    out["cells"] = cells
    if len(zr):
        d_centre = nearest_distance(zr, centres) / pitch
        d_null = nearest_distance(zr, nulls) / pitch
        out.update({
            "reflected_median_dist_to_stage_centre_over_pitch": float(np.median(d_centre)),
            "reflected_median_dist_to_null_over_pitch": float(np.median(d_null)),
            "reflected_fraction_closer_to_centre_than_null": float(np.mean(d_centre < d_null)),
            "reflected_fraction_pitch70": float(np.mean(refl["reflected_pitch_deg"] >= 45.0)),
        })
        if design.get("full_map") is not None:
            b_end = interpolate_map(design["full_map"], refl["reflected_r_m"], zr)
            b_start = interpolate_map(design["full_map"], refl["reflected_launch_r_m"], refl["reflected_launch_z_m"])
            ratio = b_end / b_start
            # a magnetic mirror needs |B|_turn / |B|_launch = 1/sin^2(pitch): 1.13 (70 deg), 8.5 (20 deg)
            needed = 1.0 / np.sin(np.radians(refl["reflected_pitch_deg"])) ** 2
            out.update({
                "reflected_median_b_turn_over_b_launch": float(np.median(ratio)),
                "reflected_fraction_b_turn_below_launch": float(np.mean(ratio < 1.0)),
                "reflected_fraction_meeting_mirror_condition": float(np.mean(ratio >= 0.95 * needed)),
            })
    if len(zw):
        d_centre = nearest_distance(zw, centres) / pitch
        d_null = nearest_distance(zw, nulls) / pitch
        out.update({
            "wall_hit_median_dist_to_null_over_pitch": float(np.median(d_null)),
            "wall_hit_fraction_closer_to_null_than_centre": float(np.mean(d_null < d_centre)),
        })
    return out


# ---- reporting --------------------------------------------------------------


def fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isnan(x):
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def mm(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{1e3 * x:.3f}"


def print_report(results: list[dict], reflections: dict, v3_state) -> None:
    short = {r["design_id"]: r["design_id"].replace("l1a-gs-v2-", "v2-")[:6] if r["design_id"].startswith("l1a") else "P2" for r in results}
    print("\n### Table A - PPM axis-field fit (window: first to last stage centre)\n")
    print("| design | stages | pitch L geom (mm) | L fit (mm) | L fit / L geom | b1 (T) | b3/b1 | b5/b1 | RMS/peak | pts per pitch | magnet fraction | r_m / L | x_w = pi r_w / L |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(f"| {short[r['design_id']]} | {r['stage_count']} | {mm(r['geometric_pitch_m'])} | {mm(r['fitted_pitch_m'])} | {fmt(r['pitch_ratio'], 4)} | {fmt(r['b1_t'])} | {fmt(r['b3_over_b1'])} | {fmt(r['b5_over_b1'])} | {fmt(r['fit_rms_over_peak'])} | {fmt(r['points_per_pitch'])} | {fmt(r['magnet_axial_fraction'])} | {fmt(r['magnet_inner_radius_over_pitch'])} | {fmt(r['x_w'])} |")

    print("\n### Table B - cusp positions: stage gap (pole plane) vs recorded axis null vs fitted-series null vs topology-v3 wall intersection z_c\n")
    print("| design | cusp | stage gap (mm) | recorded axis null (mm) | null - gap (mm) | fitted null (mm) | fitted - recorded (mm) | v3 z_c (mm) | z_c - null (mm) | v3 separatrix angle to wall normal (deg) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        for i, c in enumerate(r["cusps"], start=1):
            gap, rec, fit, v3 = c["gap_m"], c["recorded_null_m"], c["fitted_null_m"], c["v3_z_c_m"]
            print(f"| {short[r['design_id']]} | {i} | {mm(gap)} | {mm(rec)} | {mm(rec - gap)} | {mm(fit)} | {mm(fit - rec)} | {mm(v3)} | {mm(v3 - rec) if not math.isnan(v3) else '-'} | {fmt(c['v3_angle_deg'])} |")

    print("\n### Table C - wall B_r at the sampled wall radius: paraxial and Bessel extensions of the fitted axis series vs the recorded maximum\n")
    print("| design | r_w sampled (mm) | recorded max wall B_r (T) | paraxial -(r/2)dBz/dz max (T) | paraxial / recorded | fundamental b1 I1(kappa r) max (T) | fundamental / recorded | axis-fit harmonics k=1,3,5 extended (T) | extended / recorded | I1(x_w) | I1/I0(x_w) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        rec = r["recorded_wall_br_max_t"]
        print(f"| {short[r['design_id']]} | {mm(r['wall_sampled_r_m'])} | {fmt(rec)} | {fmt(r['paraxial_wall_br_max_t'])} | {fmt(r['paraxial_wall_br_max_t'] / rec)} | {fmt(r['fundamental_wall_br_max_t'])} | {fmt(r['fundamental_wall_br_max_t'] / rec)} | {fmt(r['bessel_wall_br_max_t'])} | {fmt(r['bessel_wall_br_max_t'] / rec)} | {fmt(r['I1_xw'])} | {fmt(r['I1_over_I0_xw'])} |")

    print("\n### Table C2 - harmonic content measured at the wall (Bessel series fitted to the recorded wall B_r at the axis-fitted L, z0, phi) and cusp sharpness\n")
    print("| design | wall b3 / wall b1 | wall b5 / wall b1 | wall-fitted b1 / axis-fitted b1 (Laplace consistency, 1 expected) | wall-fit RMS / max | recorded abs(B_r) FWHM at cusp 1 (mm) | pure-fundamental FWHM 2L/3 (mm) | FWHM ratio |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        h = r["wall_harmonics"]
        print(f"| {short[r['design_id']]} | {fmt(h['wall_b3_over_wall_b1'])} | {fmt(h['wall_b5_over_wall_b1'])} | {fmt(h['wall_b1_over_axis_b1'])} | {fmt(h['wall_fit_rms_over_max'])} | {mm(r['wall_fwhm_recorded_m'])} | {mm(r['wall_fwhm_fundamental_m'])} | {fmt(r['wall_fwhm_recorded_m'] / r['wall_fwhm_fundamental_m'])} |")

    print("\n### Table D - mirror ratios at the interior cusps (recorded fields) and wall-field angle\n")
    print("| design | cusp z_c (mm) | wall B at cusp (T) | recorded wall-field angle to wall normal at z_c (deg) | wall cusp B / adjacent axis peak (Koch ratio, axis form; theory I1(x_w)) | wall cusp B / wall B at stage centre (theory I1/I0(x_w)) | v3 axis_mirror_ratio (cells) | v3 wall_mirror_ratio (cells) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        v3a = ", ".join(fmt(x) for x in r["v3_axis_mirror_ratios"]) if r["v3_axis_mirror_ratios"] else "-"
        v3w = ", ".join(fmt(x) for x in r["v3_wall_mirror_ratios"]) if r["v3_wall_mirror_ratios"] else "-"
        for c in r["cusp_fields"]:
            print(f"| {short[r['design_id']]} | {mm(c['z_c_m'])} | {fmt(c['recorded_wall_b_t'])} | {fmt(c['angle_recorded_deg'])} | {fmt(c['ratio_wall_cusp_over_axis_peak'])} ({fmt(r['I1_xw'])}) | {fmt(c['ratio_wall_cusp_over_wall_centre'])} ({fmt(r['I1_over_I0_xw'])}) | {v3a} | {v3w} |")

    print("\n### Table E - launch field lines (fitted series) from the central stage centre toward the next cusp\n")
    print("| design | launch r / r_w | reaches wall before cusp | wall hit at fraction of centre-to-cusp distance | B at launch (T) | B at wall hit (T) | B_wall / B_launch | max B along line / B_launch |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        for l in r["field_lines"]:
            if l["reaches_wall"]:
                print(f"| {short[r['design_id']]} | {fmt(l['launch_r_over_rw'])} | yes | {fmt(l['wall_hit_fraction_to_cusp'])} | {fmt(l['b_launch_t'])} | {fmt(l['b_wall_t'])} | {fmt(l['ratio_wall_over_launch'])} | {fmt(l['max_along_line_over_launch'])} |")
            else:
                print(f"| {short[r['design_id']]} | {fmt(l['launch_r_over_rw'])} | no | - | - | - | - | - |")

    print("\n### Table F - electron adiabaticity numbers (launch energies of the orbit campaigns; total speed used)\n")
    print("| design | E (eV) | r_L at axis peak b1 (mm) | r_L at wall cusp field (mm) | field period 2L / lambda_c | Mendel alpha = (2L/lambda_c)^2/8 | non-adiabatic radius sqrt(m v/(e b1 kappa)) (mm) | wall-cusp scale B/abs(dBz/dz) (mm) | epsilon = r_L / scale |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        for e in r["electrons"]:
            print(f"| {short[r['design_id']]} | {fmt(e['energy_ev'])} | {mm(e['r_l_at_axis_peak_m'])} | {mm(e['r_l_at_wall_cusp_m'])} | {fmt(e['period_over_lambda_c'])} | {fmt(e['mendel_alpha'])} | {mm(e['nonadiabatic_radius_m'])} | {mm(e['wall_cusp_scale_length_m'])} | {fmt(e['epsilon_wall_cusp'])} |")

    print("\n### Table G - where the recorded reflections happened (geometry screening v1, accepted-2N, 512 orbits per design)\n")
    print("| design | reflected | wall hits | mu variation median | reflected: median distance to nearest stage centre (pitch) | to nearest axis null (pitch) | fraction closer to a stage centre than to a null | fraction with pitch angle 70 deg | median B(turn)/B(launch) | fraction with B(turn) < B(launch) | fraction meeting the mirror condition | wall hits: median distance to nearest null (pitch) | wall hits closer to null than centre |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        s = reflections.get(r["design_id"])
        if not s:
            continue
        print(f"| {short[r['design_id']]} | {s['reflected']} | {s['wall_hits']} | {fmt(s['mu_median'])} | {fmt(s.get('reflected_median_dist_to_stage_centre_over_pitch'))} | {fmt(s.get('reflected_median_dist_to_null_over_pitch'))} | {fmt(s.get('reflected_fraction_closer_to_centre_than_null'))} | {fmt(s.get('reflected_fraction_pitch70'))} | {fmt(s.get('reflected_median_b_turn_over_b_launch'))} | {fmt(s.get('reflected_fraction_b_turn_below_launch'))} | {fmt(s.get('reflected_fraction_meeting_mirror_condition'))} | {fmt(s.get('wall_hit_median_dist_to_null_over_pitch'))} | {fmt(s.get('wall_hit_fraction_closer_to_null_than_centre'))} |")
    print("\n### Table G2 - reflections per launch cell against the cell's position in the PPM field\n")
    print("| design | cell | launch z (mm) | distance to nearest axis null (pitch) | distance to nearest stage centre (pitch) | axis abs(B_z) at launch z / axis peak | orbits | reflected | wall hits |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        s = reflections.get(r["design_id"])
        if not s:
            continue
        for c in s["cells"]:
            print(f"| {short[r['design_id']]} | {c['cell_id'].replace('gs1-', '')} | {mm(c['launch_z_m'])} | {fmt(c['dist_to_null_over_pitch'])} | {fmt(c['dist_to_centre_over_pitch'])} | {fmt(c['axis_b_at_launch_over_peak'])} | {c['orbits']} | {c['reflected']} | {c['wall_hit']} |")
    if v3_state is not None:
        print(f"\nTopology catalogue used: {v3_state}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology-v3-root", type=Path, default=None, help="checkout containing modern/experiments/cusp_topology_search_v3_1 (or _v3)/results; default: this repository")
    parser.add_argument("--json", type=Path, default=None, help="write the full result to this path (outside experiments/)")
    parser.add_argument("--no-p2", action="store_true")
    args = parser.parse_args()
    if args.json is not None and (MODERN / "experiments") in args.json.resolve().parents:
        parser.error("--json must not point inside modern/experiments (artifacts are read-only)")

    v3 = load_topology_v3(args.topology_v3_root)
    designs = [load_sweep_representative(c) for c in REPRESENTATIVES]
    if not args.no_p2:
        p2 = load_p2()
        if p2 is not None:
            p2["recorded_axis_nulls_m"] = sampled_axis_nulls(p2["axis_z_m"], p2["axis_bz_t"])
            designs.append(p2)
    results, reflections = [], {}
    for design in designs:
        entry = v3.get("entries", {}).get(design["design_id"].split(" ")[0]) if v3 else None
        if entry is None and v3 and design["design_id"].startswith("divergent"):
            entry = v3["entries"].get("divergent-exit-stack")
        analysis = analyse(design, entry)
        analysis["v3_state"] = v3.get("state") if v3 else None
        results.append(analysis)
        summary = reflection_summary(design, analysis, load_reflections(design["design_id"]))
        if summary:
            reflections[design["design_id"]] = summary
    topology_note = (
        f"{v3['experiment']} ({v3.get('experiment_id')}), recorded status {v3.get('state')}" if v3 else None
    )
    print_report(results, reflections, topology_note)
    if args.json is not None:
        payload = {"results": results, "reflections": reflections, "topology_catalogue": topology_note}
        args.json.write_bytes(json.dumps(payload, indent=1, sort_keys=True, default=float).encode("utf-8").replace(b"\r\n", b"\n"))
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
