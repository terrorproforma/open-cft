"""Generate the standalone PIC-2D CFT plume dashboard (model v2.0).

Headline: the v2.0 plume development run of ``modern/experiments/pic2d_cft_plume_v1``
(L-shaped channel + plume domain, cathode emission region, two-zone neutrals, thrust
from the far-field momentum flux with the momentum-balance and Maxwell-stress closure).
Every embedded input is hash-verified against its ``.sha256.json`` sidecar and the
run's recorded protocol hash (fail-closed on protocol drift).  Panels: full-domain maps
with the thruster body, front face, cathode annulus and far-field boundary drawn (with
the shared sampling mask / binning / linear-log controls), sample cold-ion trajectories
in the window-averaged field, j_i(theta) on the far-field arc, the IEDF, the axis
potential with the acceleration region, thrust time series with the closure check, the
plume-boundary gate history and the performance numbers with their claim boundary.
Identical inputs give identical bytes; the page is self-contained and states its
development claim boundary on every view.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from math import pi, sqrt
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

MODERN = Path(__file__).resolve().parents[1]
SRC = MODERN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cft_revival.pic2d.models import ChannelGeometry, Grid2D  # noqa: E402


def _load(name: str, module_name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


steady_dashboard = _load("generate_pic2d_cft_steady_state.py", "pic2d_steady_state_generator")
snapshot_dashboard = steady_dashboard.snapshot_dashboard

EXPERIMENT = MODERN / "experiments" / "pic2d_cft_plume_v1"
RESULTS = EXPERIMENT / "results"
PROTOCOL = EXPERIMENT / "protocol.json"
CHANNEL_REFERENCE = MODERN / "experiments" / "pic2d_cft_steady_state_v2" / "results" / "summary.json"
DEFAULT_OUTPUT = Path(__file__).with_name("pic2d-cft-plume.html")
SCHEMA = "cft-pic2d-cft-plume-visualization/0.1.0"
STOP_REASONS = steady_dashboard.STOP_REASONS
_round = snapshot_dashboard._round
_matrix = snapshot_dashboard._matrix
_file_sha256 = snapshot_dashboard._file_sha256
_decimate = steady_dashboard._decimate

E_CHARGE = 1.602176634e-19
XENON_MASS_KG = 2.1801714e-25
TRAJECTORY_COUNT = 12
TRAJECTORY_MIN_SEPARATION_M = 1.0e-3
TRAJECTORY_MAX_STEPS = 20000
TRAJECTORY_POINTS = 160

# model-to-model / different-device context, never a validation gate (review blocker 6; campaign proposal Section 8)
LITERATURE_CONTEXT = [
    {"source": "Brandt et al. 2016 (Trans. JSASS Aerospace Tech. Japan 14, Pb_235)", "device": "DLR/Airbus micro-HEMPT, channel 14 x 1.5 mm, 400 V, 0.27 sccm Xe; PIC-MCC with static DSMC neutrals, Bohm diffusion, SEE, wall recycling; plume box 20 x 5 mm ('still too small' for plume ratios)",
     "numbers": "anode electron current 4.3 mA (4.5 measured); net ionisation 24 %; beam 2.5 mA (3.1 measured); plume peak 50 deg (60 measured); ~10 V / ~5 V steps at the internal cusps, main drop at the exit cusp", "label": "different closure, different geometry"},
    {"source": "Keller et al. 2015 (downscaled HEMPT family, experiment)", "device": "same micro-HEMPT family, up to 600 V", "numbers": "thrust 50-360 uN, Isp 230-860 s; beam profile and ion acceleration versus geometry", "label": "experiment on a different device; no PIC"},
    {"source": "Koch et al. IEPC-2011-236 (HEMP-T 3050)", "device": "flight-class HEMP-T, kW class", "numbers": "ion-energy peak ~ U_a - 15 V (flat interior potential, one exit drop): the validation-v1 observable named by the review", "label": "different device class; the IEDF peak - U_a below is reported as context"},
]


def _plume_grid(summary: Mapping[str, Any]) -> tuple[Grid2D, dict[str, Any]]:
    grid, grid_dict = steady_dashboard._grid(summary)
    if not grid.geometry.has_plume:
        raise ValueError("the plume dashboard needs a v2.0 plume geometry (plume_radius_m / plume_length_m)")
    return grid, grid_dict


def body_outline(grid: Grid2D) -> dict[str, Any]:
    """Thruster body, front face, anode, far-field and cathode segments in (z, r) for the map overlay."""

    geometry = grid.geometry
    z_faces = grid.z_m[: grid.axial_cells + 1]
    channel_cells = int(round(geometry.channel_length_m / grid.dz_m))
    profile: list[list[float]] = [[float(z_faces[0]), float(geometry.bore_radius_m)]]
    for j in range(channel_cells):   # stair-step wall of the plasma cell mask (mesh rule: outer radius <= wall radius at the low-z face)
        r_wall = float(geometry.wall_radius_m(z_faces[j]))
        r_cells = int(np.floor(r_wall / grid.dr_m + 1e-9))
        r_mask = r_cells * grid.dr_m
        if profile[-1][1] != r_mask:
            profile.append([float(z_faces[j]), float(r_mask)])
        profile.append([float(z_faces[j + 1]), float(r_mask)])
    z_exit = float(geometry.channel_length_m)
    return {
        "wall_profile_zr_m": [[float(f"{z:.6g}"), float(f"{r:.6g}")] for z, r in profile],
        "z_exit_m": z_exit,
        "z_max_m": float(geometry.domain_z_max_m),
        "r_bore_m": float(geometry.bore_radius_m),
        "r_exit_m": float(geometry.exit_radius_m),
        "r_body_dielectric_m": float(geometry.body_dielectric_radius_m),
        "r_plume_m": float(geometry.plume_radius_m),
        "channel_length_m": z_exit,
        "plume_length_m": float(geometry.plume_length_m),
    }


def sample_ion_trajectories(phi: np.ndarray, ionization: np.ndarray, masks: Any, grid: Grid2D, count: int = TRAJECTORY_COUNT) -> dict[str, Any]:
    """Cold test ions started at rest from the most strongly ionising cells, pushed through the window-averaged E field.

    Post-processing on the time-averaged potential map (bilinear E, leapfrog, adaptive step of half a cell,
    B ignored: the xenon gyroradius at these fields is metres against a centimetre box).  These are NOT tracked
    particles of the run; they show where the mean field sends an ion born at rest, which is the acceleration
    picture a reader wants next to the potential map.
    """

    geometry = grid.geometry
    plasma = masks.plasma_node
    dr, dz = grid.dr_m, grid.dz_m
    phi_filled = np.where(plasma, phi, np.nan)
    # one-sided differences at the plasma edge: use the plasma-only gradient and zero the field outside
    with np.errstate(invalid="ignore"):
        e_r = -np.gradient(np.nan_to_num(phi_filled, nan=0.0), dr, axis=0)
        e_z = -np.gradient(np.nan_to_num(phi_filled, nan=0.0), dz, axis=1)
    e_r = np.where(plasma, e_r, 0.0)
    e_z = np.where(plasma, e_z, 0.0)
    nr_nodes, nz_nodes = phi.shape
    source = np.where(plasma, np.nan_to_num(ionization, nan=0.0), 0.0)
    order = np.argsort(source.ravel())[::-1]
    starts: list[tuple[float, float]] = []
    for flat in order:
        if source.ravel()[flat] <= 0.0 or len(starts) >= count:
            break
        i, j = np.unravel_index(int(flat), source.shape)
        r0, z0 = float(grid.r_m[i]), float(grid.z_m[j])
        if all(sqrt((r0 - r1) ** 2 + (z0 - z1) ** 2) >= TRAJECTORY_MIN_SEPARATION_M for r1, z1 in starts):
            starts.append((r0, z0))

    def field_at(r: float, z: float) -> tuple[float, float] | None:
        fi, fj = r / dr, (z - geometry.z_min_m) / dz
        i, j = int(np.floor(fi)), int(np.floor(fj))
        if i < 0 or j < 0 or i >= nr_nodes - 1 or j >= nz_nodes - 1 or not masks.plasma_cell[i, j]:
            return None
        wi, wj = fi - i, fj - j
        w = ((1 - wi) * (1 - wj), wi * (1 - wj), (1 - wi) * wj, wi * wj)
        idx = ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
        return (sum(wk * e_r[a, b] for wk, (a, b) in zip(w, idx)), sum(wk * e_z[a, b] for wk, (a, b) in zip(w, idx)))

    v_min = sqrt(2.0 * E_CHARGE * 1.0 / XENON_MASS_KG)   # 1 eV xenon ion: the step floor
    accel = E_CHARGE / XENON_MASS_KG
    tracks = []
    for r0, z0 in starts:
        r, z, vr, vz = r0, z0, 0.0, 0.0
        points = [(r, z)]
        end = "max_steps"
        for _ in range(TRAJECTORY_MAX_STEPS):
            field = field_at(r, z)
            if field is None:
                end = "left the plasma region"
                break
            speed = max(sqrt(vr * vr + vz * vz), v_min)
            dt = 0.5 * min(dr, dz) / speed
            vr += accel * field[0] * dt
            vz += accel * field[1] * dt
            r += vr * dt
            z += vz * dt
            if r < 0.0:   # crossed the axis: reflect (axisymmetry)
                r, vr = -r, -vr
            points.append((r, z))
            if z >= geometry.domain_z_max_m or r >= geometry.plume_radius_m:
                end = "far field"
                break
            if z >= geometry.channel_length_m and r > geometry.exit_radius_m and points[-2][1] < geometry.channel_length_m:
                end = "front face"
                break
        pts = np.asarray(points)
        stride = max(1, -(-len(pts) // TRAJECTORY_POINTS))
        kept = np.vstack([pts[::stride], pts[-1:]]) if (len(pts) - 1) % stride else pts[::stride]
        energy_ev = 0.5 * XENON_MASS_KG * (vr * vr + vz * vz) / E_CHARGE
        tracks.append({"start_zr_m": [float(f"{z0:.6g}"), float(f"{r0:.6g}")], "end": end, "final_energy_ev": float(f"{energy_ev:.5g}"),
                       "final_angle_deg": float(f"{np.degrees(np.arctan2(vr, vz)) if vz or vr else 0.0:.4g}"),
                       "zr_m": [[float(f"{zz:.6g}"), float(f"{rr:.6g}")] for rr, zz in kept]})
    return {
        "method": "cold test ions at rest from the strongest ionisation cells (>= 1 mm apart), leapfrog in the window-averaged "
                  "bilinear E field with half-cell adaptive steps, B ignored (xenon gyroradius >> box), reflected at the axis; "
                  "post-processing of the mean field, not tracked particles of the run",
        "count": len(tracks),
        "tracks": tracks,
    }


def axis_profile(maps: Mapping[str, np.ndarray], masks: Any, grid: Grid2D, stride: int) -> dict[str, Any]:
    plasma = masks.plasma_node
    phi = np.where(plasma, maps["phi_v"], np.nan)[0]
    n_e = np.where(plasma, maps["n_e_per_m3"], np.nan)[0]
    t_e = np.where(plasma, maps["t_e_ev"], np.nan)[0]
    z = grid.z_m
    return {"z_m": _round(z[::stride]), "phi_v": _round(phi[::stride]), "n_e_per_m3": _round(n_e[::stride]), "t_e_ev": _round(t_e[::stride])}


def histograms(maps: Mapping[str, np.ndarray], grid: Grid2D) -> dict[str, Any]:
    nz = grid.axial_cells
    z_mid = grid.z_m[:-1] + 0.5 * grid.dz_m
    theta_edges = np.asarray(maps["plume_theta_edges_deg"], dtype=np.float64)
    iedf_edges = np.asarray(maps["iedf_edges_ev"], dtype=np.float64)
    counts = np.asarray(maps["plume_ion_counts_per_theta"], dtype=np.float64)
    cumulative = np.cumsum(counts) / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
    return {
        "theta_centres_deg": _round(0.5 * (theta_edges[:-1] + theta_edges[1:])),
        "ion_current_per_sr_a": _round(maps["plume_ion_current_per_sr_a"]),
        "ion_counts_per_theta": _round(counts),
        "ion_current_cumulative_fraction": _round(cumulative),
        "iedf_centres_ev": _round(0.5 * (iedf_edges[:-1] + iedf_edges[1:])),
        "iedf_counts": _round(maps["iedf_ion_counts"]),
        "side_z_m": _round(z_mid[:nz]),
        "side_ion_current_density_a_per_m2": _round(maps["side_ion_current_density_a_per_m2"]),
        "side_electron_current_density_a_per_m2": _round(maps["side_electron_current_density_a_per_m2"]),
    }


def channel_reference(path: Path = CHANNEL_REFERENCE) -> dict[str, Any] | None:
    """Hash-verified digest of the channel-only v1.3 plateau (Dirichlet exit plane) as the same-code reference."""

    if not path.is_file():
        return None
    snapshot_dashboard._verify_sidecar(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    currents = summary.get("window_currents_a") or {}
    neutral = summary.get("neutral_inventory") or {}
    return {
        "experiment_id": summary["experiment_id"], "model_version": summary["model_version"], "case_id": summary["case"]["id"],
        "summary_sha256": _file_sha256(path),
        "discharge_a": currents.get("discharge_a"), "exit_ion_beam_a": currents.get("exit_ion_beam_a"),
        "ionization_rate_per_s": currents.get("ionization_rate_per_s"),
        "utilisation": neutral.get("propellant_utilisation_trailing"),
        "phi_max_v": summary["window_maps_summary"].get("phi_max_v"),
        "n_e_peak_per_m3": summary["window_maps_summary"].get("n_e_peak_per_m3"),
        "note": "channel-only v1.3 plateau with the exit plane at Dirichlet 0 V and exit-plane electron injection: the closure the "
                "v2.0 plume block replaces; same code and operating point, different boundary model",
    }


def build_payload(results: Path = RESULTS, protocol_path: Path = PROTOCOL, reference_path: Path | None = CHANNEL_REFERENCE) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw: dict[str, Any] = {}
    case = steady_dashboard.build_case(results, protocol_path, label=f"{protocol['model_version']} plume development run", role="headline", raw_out=raw)
    summary, maps, series = raw["summary"], raw["maps"], raw["series"]
    grid, grid_dict = _plume_grid(summary)
    masks = raw["masks"]
    nz = grid.axial_cells
    stride = 1 if nz <= 256 else 2
    if not case["cusps"]["cusp_z_m"]:   # the P2 bicubic authority covers the channel only: locate the cusps on the channel grid
        geometry = grid.geometry
        channel = Grid2D(ChannelGeometry(geometry.bore_radius_m, geometry.z_min_m, geometry.z_max_m, geometry.cone_start_z_m, geometry.exit_radius_m),
                         int(round(geometry.exit_radius_m / grid.dr_m)), int(round(geometry.channel_length_m / grid.dz_m)))
        case["cusps"] = steady_dashboard.cusp_positions(maps, channel, grid_dict)
    plume_keys = [k for k in series if k.startswith(("momentum_", "plume_", "peak_node_")) or k in ("current_cathode_emission_a", "current_body_face_electron_a",
                                                                                     "current_body_face_ion_a", "current_plume_ionization_rate_per_s",
                                                                                     "current_anode_ion_a", "current_anode_electron_a")]
    case["plume_series"] = {key: _round(_decimate(series[key], case["series_stride"])) for key in plume_keys}
    case["plume"] = summary["plume"]
    case["histograms"] = histograms(maps, grid)
    case["axis"] = axis_profile(maps, masks, grid, stride)
    case["body"] = body_outline(grid)
    case["trajectories"] = sample_ion_trajectories(np.asarray(maps["phi_v"]), np.asarray(maps["ionization_rate_per_m3_s"]), masks, grid)
    case["v2_0_options"] = summary["provenance"].get("v2_0_options")
    case["v1_4_options"] = summary["provenance"].get("v1_4_options")
    case["grid_heating_triad"] = summary.get("grid_heating_triad")
    case["peak_node_debye"] = summary.get("peak_node_debye")
    case["config"]["cathode"] = summary["provenance"]["config"].get("cathode")
    case["config"]["plume_boundary_gate"] = summary["provenance"]["config"].get("plume_boundary_gate")
    case["config"]["geometry"] = grid_dict["geometry"]
    case["mesh"] = summary["provenance"]["mesh"]
    plume = summary["plume"] or {}
    domain = protocol.get("domain_v2_0", {})
    payload = {
        "schema": SCHEMA,
        "experiment_id": protocol["experiment_id"],
        "model_version": protocol["model_version"],
        "status": protocol["status"],
        "claim_boundary": protocol.get("claim_boundary"),
        "claim_statement": (
            f"Development plume run (model {protocol['model_version']}): the channel of the v1.3/v1.4 development runs with a "
            f"{fmt_mm(grid.geometry.plume_radius_m)} x {fmt_mm(grid.geometry.plume_length_m)} mm plume box at the same operating point. "
            "Not preregistered. Not validated against any experiment. Not a thruster performance prediction: the thrust, specific "
            "impulse, anode efficiency, divergence and IEDF below are development numbers bound to this box size (Dirichlet 0 V far "
            "field, Brandt et al. 2016 found a 20 x 5 mm box still too small), to the volumetric cathode model, to the two-zone "
            "neutrals without ion-neutral collisions and to the particle resolution (~5 % band from the channel-only convergence "
            "pair). The momentum-flux thrust and the momentum-balance thrust are two estimates of one quantity from one run; their "
            "closure is a conservation check of the code, not an accuracy statement about the device. Numerics verified by the tests "
            "in modern/tests/pic2d; physics simplified as listed."
        ),
        "simplifications": protocol["simplifications"],
        "protocol": {
            "file_sha256": _file_sha256(protocol_path),
            "model_spec": protocol.get("model_spec"),
            "operating_point": protocol["operating_point"],
            "numerics": protocol["numerics"],
            "geometry": protocol["geometry"],
            "cathode": protocol.get("cathode"),
            "domain_v2_0": domain,
            "stopping_rule": protocol["stopping_rule"],
            "budget_expectations": protocol.get("budget_expectations"),
        },
        "performance": {
            "thrust_total_n": plume.get("thrust_total_n"), "thrust_flux_n": plume.get("thrust_flux_n"), "cold_gas_thrust_n": plume.get("cold_gas_thrust_n"),
            "thrust_balance_n": plume.get("thrust_balance_n"), "closure_fraction": plume.get("closure_fraction"),
            "electrostatic_force_thruster_n": plume.get("electrostatic_force_thruster_n"),
            "specific_impulse_s": plume.get("specific_impulse_s"), "anode_efficiency": plume.get("anode_efficiency"),
            "divergence_half_angle_95_deg": plume.get("divergence_half_angle_95_deg"), "iedf_mean_energy_ev": plume.get("iedf_mean_energy_ev"),
            "iedf_peak_energy_ev": plume.get("iedf_peak_energy_ev"), "iedf_peak_minus_anode_v": plume.get("iedf_peak_minus_anode_v"),
            "exit_plane_axis_potential_v": plume.get("exit_plane_axis_potential_v"), "acceleration_z90_m": plume.get("acceleration_z90_m"),
            "acceleration_z10_m": plume.get("acceleration_z10_m"), "acceleration_width_m": plume.get("acceleration_width_m"),
            "mass_flow_kg_per_s": plume.get("mass_flow_kg_per_s"), "discharge_a": (summary.get("window_currents_a") or {}).get("discharge_a"),
            "cathode_emission_a": (summary.get("window_currents_a") or {}).get("cathode_emission_a"),
            "window_step_range": plume.get("window_step_range"), "window_samples": plume.get("window_samples"),
        },
        "literature_context": LITERATURE_CONTEXT,
        "channel_reference": channel_reference(reference_path) if reference_path is not None else None,
        "cases": [case],
    }
    validate_payload(payload)
    return payload


def fmt_mm(value: float) -> str:
    return f"{value * 1e3:g}"


def validate_payload(payload: Mapping[str, Any]) -> None:
    required = {
        "schema", "experiment_id", "model_version", "status", "claim_boundary", "claim_statement", "simplifications", "protocol",
        "performance", "literature_context", "channel_reference", "cases",
    }
    if set(payload) != required:
        raise ValueError("payload keys do not match the closed schema")
    if payload["schema"] != SCHEMA:
        raise ValueError("unsupported payload schema")
    if payload["status"] != "development_screening_not_preregistered":
        raise ValueError("payload must carry the development/screening status")
    statement = payload["claim_statement"].lower()
    for phrase in ("not preregistered", "not validated", "development numbers", "not a thruster performance prediction"):
        if phrase not in statement:
            raise ValueError(f"claim boundary must state '{phrase}'")
    if not payload["simplifications"]:
        raise ValueError("simplifications must be listed")
    if len(payload["cases"]) != 1 or payload["cases"][0]["role"] != "headline":
        raise ValueError("the plume dashboard embeds exactly one headline case")
    if not isinstance(payload["protocol"]["file_sha256"], str) or len(payload["protocol"]["file_sha256"]) != 64:
        raise ValueError("protocol file_sha256 must be a SHA-256")
    for entry in payload["literature_context"]:
        if "label" not in entry or "validation" in entry["label"].lower() and "never" not in entry["label"].lower() and "not" not in entry["label"].lower():
            raise ValueError("literature context must be labelled as context, never as validation")
    case = payload["cases"][0]
    for key in ("summary_sha256", "maps_npz_sha256", "series_npz_sha256", "protocol_sha256"):
        digest = case[key]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{key} must be a SHA-256")
    if case["protocol_sha256"] != payload["protocol"]["file_sha256"]:
        raise ValueError("case protocol hash differs from the protocol file")
    if case["stop_reason"] not in STOP_REASONS:
        raise ValueError("unknown stop reason")
    if case["plume"] is None or case["plume"].get("thrust_total_n") is None:
        raise ValueError("the plume summary block with the thrust estimates is required")
    nr, nz = len(case["grid_r_m"]), len(case["grid_z_m"])
    for key in steady_dashboard.MAP_KEYS:
        matrix = case["maps"][key]
        if len(matrix) != nr or any(len(row) != nz for row in matrix):
            raise ValueError(f"map {key} shape does not match the grid")
    snapshot_dashboard.validate_sampling(case)
    n = len(case["series"]["time_s"])
    for name in ("series", "plume_series"):
        for key, values in case[name].items():
            if len(values) != n:
                raise ValueError(f"{name} {key} length differs from time_s")
    for key in ("momentum_thrust_flux_n", "momentum_thrust_total_n", "momentum_thrust_balance_n", "momentum_closure_fraction",
                "plume_charge_fraction_of_peak", "current_cathode_emission_a"):
        if key not in case["plume_series"]:
            raise ValueError(f"plume series {key} is required")
    body = case["body"]
    if not (0.0 < body["r_bore_m"] <= body["r_exit_m"] <= body["r_body_dielectric_m"] <= body["r_plume_m"]) or not body["z_exit_m"] < body["z_max_m"]:
        raise ValueError("body outline radii/lengths are inconsistent")
    if len(body["wall_profile_zr_m"]) < 2:
        raise ValueError("wall profile must have at least two points")
    if len(case["axis"]["z_m"]) != nz:
        raise ValueError("axis profile length differs from the grid")
    h = case["histograms"]
    if len(h["theta_centres_deg"]) != len(h["ion_current_per_sr_a"]) or len(h["iedf_centres_ev"]) != len(h["iedf_counts"]):
        raise ValueError("histogram lengths are inconsistent")
    if case["trajectories"]["count"] != len(case["trajectories"]["tracks"]):
        raise ValueError("trajectory count mismatch")
    for track in case["trajectories"]["tracks"]:
        if len(track["zr_m"]) < 1 or track["end"] not in ("far field", "front face", "left the plasma region", "max_steps"):
            raise ValueError("trajectory record malformed")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PIC-2D CFT plume (development, model v2.0)</title>
<style>
:root{color-scheme:dark;--bg:#07100f;--panel:#0f1c1a;--panel2:#14262380;--text:#eef7f4;--muted:#9bb8b0;--line:#2b4540;--accent:#5ad6c0;--warn:#ffcf67;--red:#ff6b6b;--blue:#58a8ff;--shadow:#0008;--window:#5ad6c022}
[data-theme=light]{color-scheme:light;--bg:#edf5f2;--panel:#fff;--panel2:#f2f8f6;--text:#10231f;--muted:#4f6a63;--line:#bfd3cc;--accent:#087f6e;--warn:#7a5700;--red:#b83232;--blue:#176db5;--shadow:#3452;--window:#087f6e22}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 0,#153b34 0,transparent 36rem),var(--bg);color:var(--text);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
button,select,input{font:inherit;color:var(--text);background:var(--panel2);border:1px solid var(--line);border-radius:.55rem;padding:.48rem .7rem}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,canvas:focus-visible,input:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
header,main,footer{width:min(1500px,calc(100% - 2rem));margin:auto}header{padding:2rem 0 1rem}.eyebrow{color:var(--accent);font-weight:750;letter-spacing:.14em;text-transform:uppercase}h1{font-size:clamp(1.9rem,4.5vw,3.8rem);line-height:.98;margin:.2rem 0 .8rem;max-width:960px}h2{margin:.1rem 0 .8rem;font-size:1.1rem}p{margin:.35rem 0}
.claim{border:1px solid #8b681c;background:#513d1438;color:var(--warn);padding:.8rem 1rem;border-radius:.65rem;font-weight:650}.claim ul{margin:.4rem 0 0 1.1rem;font-weight:500;color:var(--text)}
.controls{display:flex;gap:.6rem;flex-wrap:wrap;align-items:end;margin:1rem 0}.control{display:grid;gap:.25rem}.control label{color:var(--muted);font-size:.8rem}.check{display:flex;gap:.4rem;align-items:center;padding:.48rem 0}
.grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(280px,.8fr);gap:1rem;margin:1rem 0}.panel{background:linear-gradient(145deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:.9rem;padding:1rem;box-shadow:0 12px 30px var(--shadow);min-width:0}
.canvas-wrap{position:relative;min-height:300px}.canvas-wrap canvas{width:100%;height:clamp(320px,40vw,560px);display:block}.tip{position:absolute;pointer-events:none;background:#07100fee;color:#fff;border:1px solid #7f9a93;border-radius:.35rem;padding:.35rem .5rem;display:none;white-space:nowrap}
.kv{display:grid;grid-template-columns:1fr auto;gap:.22rem .6rem}.kv span{min-width:0;overflow-wrap:anywhere}.kv span:nth-child(odd){color:var(--muted)}.kv span:nth-child(even){font-variant-numeric:tabular-nums;text-align:right}h1,h2,h3,p,li{overflow-wrap:anywhere}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.plot{width:100%;height:260px;display:block}.wide{grid-column:1/-1}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:.1rem .5rem;margin:.1rem .2rem .1rem 0;color:var(--muted)}code{font-size:.78em;overflow-wrap:anywhere}.small{font-size:.82rem;color:var(--muted)}footer{padding:1rem 0 2.5rem;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th{text-align:left;color:var(--muted);font-weight:600}td,th{padding:.15rem .4rem;border-bottom:1px solid var(--line)}.ok{color:var(--accent)}.marginal{color:var(--warn)}.bad{color:var(--red)}
.legend span{display:inline-block;margin-right:.9rem}.legend i{display:inline-block;width:.9rem;height:.6rem;margin-right:.3rem;vertical-align:middle;border-radius:2px}
@media(max-width:900px){.grid,.plots{grid-template-columns:1fr}.canvas-wrap canvas{height:380px}}@media(max-width:520px){header,main,footer{width:min(100% - 1rem,1500px)}.canvas-wrap canvas{height:300px}.panel{padding:.7rem}}
</style>
</head>
<body>
<header><div class="eyebrow">PIC-MCC · axisymmetric (r,z) · development plume run · model v2.0</div><h1>Divergent-exit CFT with a plume box: exhaust shape, thrust and its closure</h1>
<div id="claim" class="claim" role="note"></div>
<div class="controls">
<div class="control"><label for="case">Case</label><select id="case"></select></div>
<div class="control"><label for="map">Map (time-averaged over the reporting window)</label><select id="map"><option value="n_e_per_m3">Electron density n_e (m⁻³)</option><option value="n_i_per_m3">Ion density n_i (m⁻³)</option><option value="phi_v">Potential φ (V)</option><option value="t_e_ev">Electron temperature T_e (eV)</option><option value="ionization_rate_per_m3_s">Ionisation rate (m⁻³ s⁻¹)</option></select></div>
<div class="control"><label for="scale">Colour scale</label><select id="scale"><option value="linear">linear</option><option value="log">log10</option></select></div>
__MAP_CONTROLS__
<label class="check small"><input type="checkbox" id="tracks" checked> ion trajectories (mean field)</label>
<button id="theme" type="button" aria-pressed="false">Light theme</button>
</div><p class="small">Arrow keys move the map cursor; Home resets it. Overlay: hatched = thruster body (magnets/yoke, outside the plasma cell mask); red = anode; orange = dielectric front-face ring; blue = grounded front-face conductor; dotted = far-field Dirichlet boundary; dashed box = cathode emission annulus; dashed verticals = cusp planes. Thin lines: cold test-ion trajectories in the window-averaged field (post-processing, not tracked particles).</p></header>
<main>
<section class="panel" style="margin:1rem 0"><h2>Thrust, closure and performance (development numbers)</h2><div id="performance"></div></section>
<section class="grid">
<div class="panel"><h2 id="mapTitle">Full-domain map</h2><div class="canvas-wrap"><canvas id="field" tabindex="0" role="img" aria-label="Interactive (r,z) heatmap of the selected window-averaged quantity over the channel and the plume with the thruster body drawn"></canvas><div id="tip" class="tip" role="status" aria-live="polite"></div></div><p class="small" id="mapCaption"></p><p class="small">Canvas raster of the node grid (radial-major) over the L-shaped plasma region. White: outside the plasma cell mask; grey: sampled by fewer macro-particles than the threshold (the "speckle" of a log map is the counting noise of those cells, not structure). The channel wall and the cone are one-cell stair-steps of the mask; the exit lip, the front face and the far-field box are internal/outer boundaries of the same mask.</p></div>
<aside class="panel"><h2 id="detailTitle">Case details</h2><div id="details"></div></aside>
</section>
<section class="plots">
<div class="panel"><h2>Thrust estimates and closure</h2><canvas class="plot" id="thrust" role="img" aria-label="Momentum-flux thrust, total thrust with cold gas, momentum-balance thrust and Maxwell-stress force versus time"></canvas><p class="small">T_flux = Σ m v_z W/Δt of the particles leaving through the far-field boundary (minus the cathode's injected momentum); T_total adds the inventory's cold-gas effusion; T_balance = −F_on-thruster from the particle-side ledger (wall/anode/front-face impact momentum minus the field impulse on the plasma); F_es = Maxwell stress on the solid boundaries (field side). Closure = (T_flux − T_balance) / max(|T_flux|, |T_balance|) over the window; the stored-momentum rate and the collision momentum handed to the neutrals are the non-closing terms and are reported.</p></div>
<div class="panel"><h2>Closure fraction and plume-boundary gate</h2><canvas class="plot" id="closure" role="img" aria-label="Closure fraction and far-field charge fraction of the peak versus time"></canvas></div>
<div class="panel"><h2>Currents: anode, cathode, far field</h2><canvas class="plot" id="currents" role="img" aria-label="Discharge, cathode emission, beam and wall currents versus time"></canvas></div>
<div class="panel"><h2>Axis potential and acceleration region</h2><canvas class="plot" id="axis" role="img" aria-label="Potential on the axis versus z with the exit plane and the 90/10 percent acceleration region marked"></canvas></div>
<div class="panel"><h2>Ion current per solid angle j_i(θ) on the far-field arc</h2><canvas class="plot" id="theta" role="img" aria-label="Ion current per steradian versus polar angle from the aperture centre, with the 95 percent half-angle"></canvas></div>
<div class="panel"><h2>Ion energy distribution at the far-field boundary</h2><canvas class="plot" id="iedf" role="img" aria-label="Ion energy distribution at the far-field boundary with the anode potential marked"></canvas></div>
<div class="panel"><h2>Far-field current densities</h2><canvas class="plot" id="farfield" role="img" aria-label="Ion and electron current densities on the far plane versus radius and on the side cylinder versus z"></canvas></div>
<div class="panel"><h2>Macro-particle counts</h2><canvas class="plot" id="counts" role="img" aria-label="Electron and ion macro-particle counts versus time"></canvas></div>
<div class="panel"><h2>Neutral inventory (channel zone)</h2><canvas class="plot" id="neutral" role="img" aria-label="Channel neutral density and analytic fixed point versus time"></canvas></div>
<div class="panel"><h2>Energy ledger</h2><canvas class="plot" id="energy" role="img" aria-label="Kinetic, field and total energy with the interval ledger residual"></canvas></div>
<div class="panel"><h2>Stability metrics</h2><canvas class="plot" id="wpe" role="img" aria-label="Peak plasma-frequency times timestep and peak-node cells per Debye length versus time"></canvas></div>
<div class="panel"><h2>Wall impact flux along the channel dielectric</h2><canvas class="plot" id="wall" role="img" aria-label="Electron and ion wall flux versus axial position with cusp planes"></canvas></div>
</section>
<section class="panel" style="margin:1rem 0"><h2>Domain, boundaries, cathode and neutrals (model v2.0 choices)</h2><div id="model"></div></section>
<section class="panel" style="margin:1rem 0"><h2>Literature context (different closures and devices — never a validation)</h2><div id="literature"></div></section>
<section class="panel" style="margin:1rem 0"><h2>Simplifications (model v2.0) and identity</h2><div id="identity"></div></section>
</main><footer>Self-contained offline dashboard generated by <code>modern/visualization/generate_pic2d_cft_plume.py</code>. Development evidence only: single seed, not preregistered, not validated, development numbers bound to the box size and the closure.</footer>
<script id="pic2d-data" type="application/json">__DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("pic2d-data").textContent);
const $=id=>document.getElementById(id);let selected=0,mapKey="n_e_per_m3",scaleMode="linear",cursor=null,raf=0,showTracks=true;
const caseSelect=$("case");DATA.cases.forEach((c,i)=>{const o=document.createElement("option");o.value=i;o.textContent=c.label;caseSelect.append(o)});
const fmt=(v,n=4)=>v==null||!isFinite(v)?"–":Number(v).toLocaleString(undefined,{maximumSignificantDigits:n});
const sci=(v,n=3)=>v==null||!isFinite(v)?"–":Number(v).toExponential(n-1);
const pct=(v,n=3)=>v==null||!isFinite(v)?"–":fmt(v*100,n)+" %";
const uN=v=>v==null||!isFinite(v)?"–":fmt(v*1e6,4)+" µN";
$("claim").innerHTML=`<strong>Claim boundary:</strong> ${DATA.claim_statement}<ul>${DATA.simplifications.map(s=>`<li>${s}</li>`).join("")}</ul>`;
const themeColor=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function setup(canvas){const r=canvas.getBoundingClientRect(),dpr=Math.max(1,window.devicePixelRatio||1),w=Math.max(1,Math.round(r.width*dpr)),h=Math.max(1,Math.round(r.height*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}const c=canvas.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);return {c,w:r.width,h:r.height}}
function color(t,signed){t=Math.max(0,Math.min(1,t));if(signed){if(t<.5){const q=t*2;return `rgb(${Math.round(35+220*q)},${Math.round(92+163*q)},255)`}const q=(t-.5)*2;return `rgb(255,${Math.round(255-210*q)},${Math.round(255-215*q)})`}return `rgb(${Math.round(12+240*t)},${Math.round(28+190*Math.sqrt(t))},${Math.round(90+100*(1-t))})`}
__MAP_VIEW_JS__
function cls(v,good,marginal){return v==null?"":Math.abs(v)<good?"ok":Math.abs(v)<marginal?"marginal":"bad"}
function renderPerformance(){const c=DATA.cases[selected],P=DATA.performance,pl=c.plume||{},op=DATA.protocol.operating_point,g=c.config.geometry;
const rows=[["thrust T_total = T_flux + cold gas (window mean)",`<strong>${uN(P.thrust_total_n)}</strong> = ${uN(P.thrust_flux_n)} + ${uN(P.cold_gas_thrust_n)} (ions ${uN(pl.thrust_flux_ions_n)}, electrons ${uN(pl.thrust_flux_electrons_n)}, cathode-injected ${uN(pl.cathode_injected_momentum_rate_n)})`],
["momentum-balance thrust −F_on-thruster (particle ledger)",`${uN(P.thrust_balance_n)} → closure (T_flux − T_balance)/max <span class="${cls(P.closure_fraction,.1,.25)}">${pct(P.closure_fraction,3)}</span>; non-closing terms: stored-momentum rate ${uN(pl.stored_momentum_rate_n)}, collision momentum to neutrals ${uN(pl.collision_momentum_rate_n)}, far-field electrostatic force ${uN(P.electrostatic_force_thruster_n==null?null:pl.electrostatic_force_far_field_n)}`],
["Maxwell-stress force on the thruster (field side)",`${uN(P.electrostatic_force_thruster_n)} (independent of the particle ledger; first-order boundary discretisation)`],
["momentum-ledger residual (round-off check)",`max |interval residual| ${sci(pl.ledger_residual_max_kg_m_s,2)} kg m/s`],
["specific impulse T_total / (ṁ g₀)",`${fmt(P.specific_impulse_s,4)} s at ṁ = ${fmt(P.mass_flow_kg_per_s*1e6,4)} mg/s`],
["anode efficiency T² / (2 ṁ U_a I_d)",`${pct(P.anode_efficiency,3)} at I_d = ${fmt(P.discharge_a*1e3,4)} mA, U_a = ${op.anode_potential_v} V (cathode emission ${fmt(P.cathode_emission_a*1e3,4)} mA)`],
["beam divergence (95 % of the far-field ion current)",`${fmt(P.divergence_half_angle_95_deg,3)}° half-angle about the aperture centre (${fmt(pl.far_field_ion_crossings_in_window,4)} macro-ions in the window)`],
["IEDF at the far-field boundary",`mean ${fmt(P.iedf_mean_energy_ev,4)} eV, peak ${fmt(P.iedf_peak_energy_ev,4)} eV → peak − U_a = ${fmt(P.iedf_peak_minus_anode_v,3)} V (context: Koch et al. 2011 HEMP-T 3050 ≈ −15 V — different device, different closure)`],
["exit-plane axis potential (self-consistent)",`${fmt(P.exit_plane_axis_potential_v,4)} V; acceleration region (axis potential 90 % → 10 % of its drop) z = ${fmt(P.acceleration_z90_m*1e3,4)} → ${fmt(P.acceleration_z10_m*1e3,4)} mm, width ${fmt(P.acceleration_width_m*1e3,3)} mm (exit plane at ${fmt(g.z_max_m*1e3,3)} mm)`],
["reporting window",`steps ${P.window_step_range?P.window_step_range.join("–"):"–"} (${P.window_samples} series samples · ${duration(c.sampling.window_s)} of maps)`],
["plume-boundary gate (charge pile-up)",`max far-field net charge / peak n_e over the run <span class="${cls(pl.charge_fraction_of_peak_max,.1,.25)}">${pct(pl.charge_fraction_of_peak_max,3)}</span> (gate ${c.config.plume_boundary_gate?pct(c.config.plume_boundary_gate.max_charge_fraction,2):"off"})`]];
$("performance").innerHTML=`<table aria-label="Thrust, closure and performance"><tbody>${rows.map(([k,v])=>`<tr><th>${k}</th><td>${v}</td></tr>`).join("")}</tbody></table><p class="small">Development numbers: two estimates of one quantity from one run; the closure is a conservation check of the code, not an accuracy statement about the device. The thrust is bound to the box size and the far-field Dirichlet closure (block D sensitivity pending), to the volumetric cathode, to the neutral model without ion-neutral collisions and to the ~5 % particle-resolution band of the channel-only convergence pair.</p>`}
function renderDetails(){const c=DATA.cases[selected],g=c.stability_gate,op=DATA.protocol.operating_point,geo=c.config.geometry,ca=c.config.cathode||{},pl=c.plateau||{},ni=c.neutral_inventory||{},w=c.window_maps_summary,lg=c.ledger||{},pk=c.peak_node_debye||{};
let html=`<div class="kv"><span>backend</span><span>${c.backend}</span><span>domain</span><span>channel ${fmt(geo.z_max_m*1e3,3)} mm + plume ${fmt(geo.plume_radius_m*1e3,3)} × ${fmt(geo.plume_length_m*1e3,3)} mm</span><span>grid · Δr × Δz</span><span>${c.config.grid.radial_cells}×${c.config.grid.axial_cells} · ${fmt(c.config.grid.dr_m*1e6,3)} × ${fmt(c.config.grid.dz_m*1e6,3)} µm</span><span>plasma cells / unknowns</span><span>${c.mesh.plasma_cells} / ${c.mesh.unknown_nodes}</span><span>Δt · ion subcycle</span><span>${sci(c.config.dt_s,3)} s · k = ${DATA.protocol.numerics.ion_subcycle}</span><span>macro weight · seed</span><span>${sci(c.config.macro_weight,2)} · ${c.case.seed}</span><span>anode / far field</span><span>${c.config.potentials.anode_v} / ${c.config.potentials.exit_v} V</span><span>front face</span><span>dielectric to r = ${fmt(geo.body_dielectric_radius_m*1e3,3)} mm, grounded beyond</span><span>cathode annulus</span><span>r ${fmt(ca.r_inner_m*1e3,3)}–${fmt(ca.r_outer_m*1e3,3)} mm, z ${fmt(ca.z_start_m*1e3,3)}–${fmt(ca.z_end_m*1e3,3)} mm</span><span>cathode rule</span><span>${ca.current_rule} · floor ${fmt(ca.current_a*1e3,3)} mA${ca.max_current_a?` · ceiling ${fmt(ca.max_current_a*1e3,3)} mA`:""} @ ${ca.electron_temperature_ev} eV</span><span>feed</span><span>${fmt(DATA.performance.mass_flow_kg_per_s*1e6,4)} mg/s</span><span>steps / time</span><span>${c.steps_completed} · ${fmt(c.simulated_time_s*1e6,4)} µs (${fmt(c.ion_transit_times,3)} τ_i)</span><span>stop</span><span>${c.stop_reason.replaceAll("_"," ")}</span><span>plateau drifts I_d / N_e / n_g</span><span>${pct(pl.discharge_current_drift,2)} / ${pct(pl.electron_count_drift,2)} / ${pct(pl.neutral_density_drift,2)}</span><span>peak / mean n_e</span><span>${sci(w.n_e_peak_per_m3)} / ${sci(w.n_e_mean_per_m3)} m⁻³</span><span>φ range · ⟨T_e⟩_n</span><span>${fmt(w.phi_min_v,3)}…${fmt(w.phi_max_v,3)} V · ${fmt(w.t_e_density_weighted_mean_ev,3)} eV</span><span>n_g channel (window) · utilisation net / gross</span><span>${sci(ni.trailing_20pct_mean_density_per_m3,3)} · ${pct(ni.net_utilisation_trailing,3)} / ${pct(ni.gross_utilisation_trailing,3)}</span><span>peak-node cells/λ_D (trailing mean · max)</span><span>${fmt(pk.trailing_20pct_mean_cells_per_debye,3)} · ${fmt(pk.max_cells_per_debye,3)}</span><span>energy-ledger residual / electrode work</span><span>${pct(lg.cumulative_residual_over_electrode_work,3)}</span><span>wall · throughput</span><span>${fmt(c.wall_seconds_total/3600,3)} h · ${fmt(c.ms_per_step_last_session,3)} ms/step · ${c.sessions?c.sessions.length:"–"} session(s)</span><span>maps kind</span><span>${c.maps_kind||"–"}</span></div>`;
html+=`<h2 style="margin-top:1rem">Stability gate (configured reference)</h2><div class="kv"><span>ω_pe Δt</span><span>${fmt(g.omega_pe_dt,3)}</span><span>Ω_ce Δt</span><span>${fmt(g.omega_ce_dt,3)}</span><span>cell / λ_D</span><span>${fmt(g.cell_debye_ratio,3)}</span><span>Courant</span><span>${fmt(g.particle_courant,3)}</span><span>P_coll</span><span>${sci(g.max_collision_probability,2)}</span><span>max |B| on plasma nodes</span><span>${fmt(g.max_b_t*1e3,4)} mT</span></div>`;
if(c.stability_gate_message)html+=`<p class="small"><strong>Fail-closed stop:</strong> ${c.stability_gate_message}</p>`;
$("detailTitle").textContent=c.label;$("details").innerHTML=html;$("mapTitle").textContent=`${$("map").selectedOptions[0].textContent} — ${c.label}`;
const v2=c.v2_0_options||{},D=DATA.protocol.domain_v2_0||{};$("model").innerHTML=`<div class="kv" style="grid-template-columns:repeat(auto-fit,minmax(300px,1fr))"><span>domain</span><span>${v2.domain||"–"}</span><span>front face</span><span>${v2.front_face||"–"}</span><span>far field</span><span>${v2.far_field||"–"}</span><span>neutrals</span><span>${v2.neutrals||"–"}</span><span>legacy exit-plane injection</span><span>${v2.legacy_exit_plane_injection?"on":"off (cathode annulus replaces it)"}</span><span>histograms</span><span>${v2.histograms?`${v2.histograms.theta_bins} θ bins over 0–90°, ${v2.histograms.iedf_bins} IEDF bins to ${fmt(v2.histograms.iedf_max_ev,4)} eV`:"–"}</span><span>v1.4 options</span><span>${c.v1_4_options?`recycling ${c.v1_4_options.wall_recycling?"on":"off"}; relaxation ${c.v1_4_options.neutral_relaxation}; graph ${c.v1_4_options.step_graph?"on":"off"}`:"–"}</span></div>${Object.entries(D).map(([k,v])=>`<p class="small"><strong>${k.replaceAll("_"," ")}:</strong> ${typeof v==="string"?v:JSON.stringify(v)}</p>`).join("")}<p class="small">Trajectories: ${c.trajectories.method}. ${c.trajectories.count} tracks; ends: ${Object.entries(c.trajectories.tracks.reduce((a,t)=>(a[t.end]=(a[t.end]||0)+1,a),{})).map(([k,v])=>`${k} × ${v}`).join(", ")}; final energies ${c.trajectories.tracks.map(t=>fmt(t.final_energy_ev,3)).join(", ")} eV.</p>`;
const R=DATA.channel_reference;$("literature").innerHTML=`<table aria-label="Literature context"><thead><tr><th>source</th><th>device / closure</th><th>numbers</th><th>label</th></tr></thead><tbody>${DATA.literature_context.map(l=>`<tr><td>${l.source}</td><td>${l.device}</td><td>${l.numbers}</td><td><em>${l.label}</em></td></tr>`).join("")}${R?`<tr><td>this repository: ${R.experiment_id} (${R.model_version}, case ${R.case_id})</td><td>${R.note}</td><td>I_d ${fmt(R.discharge_a*1e3,4)} mA; I_beam,i ${fmt(R.exit_ion_beam_a*1e3,4)} mA; S ${sci(R.ionization_rate_per_s,3)} s⁻¹; utilisation ${pct(R.utilisation,3)}; φ_max ${fmt(R.phi_max_v,4)} V; peak n_e ${sci(R.n_e_peak_per_m3,3)}</td><td><em>same code, different boundary model (summary <code>${R.summary_sha256.slice(0,12)}</code>)</em></td></tr>`:""}</tbody></table><p class="small">These rows are model-to-model or different-device context labelled by closure (review blocker 6); the campaign proposal forbids reading them as a validation gate.</p>`;
$("identity").innerHTML=`<p><span class="badge">status</span> ${DATA.status.replaceAll("_"," ")}</p><p><span class="badge">model</span> ${DATA.model_version} (${DATA.protocol.model_spec||"–"})</p><p><span class="badge">protocol SHA-256</span> <code>${DATA.protocol.file_sha256}</code> (the run recorded this hash)</p><p><span class="badge">case summary SHA-256</span> <code>${c.summary_sha256}</code></p><p><span class="badge">maps npz SHA-256</span> <code>${c.maps_npz_sha256}</code></p><p><span class="badge">series npz SHA-256</span> <code>${c.series_npz_sha256}</code></p><p><span class="badge">git HEAD at run</span> <code>${c.git_head||"–"}</code></p><p><span class="badge">P2 field map SHA-256</span> <code>${c.field.field_map_sha256}</code> (design ${c.field.provenance.design_id||"–"})</p><p><span class="badge">cross sections</span> ${c.cross_sections?c.cross_sections.provenance_status+" · payload <code>"+c.cross_sections.payload_sha256+"</code>":"–"}</p><p><span class="badge">cusp planes</span> ${c.cusps.source}</p>`}
function bounds(w,h){return {l:58,t:18,r:w-78,b:h-46}}
function mapPoint(z,r,c,b){const zs=c.grid_z_m,rs=c.grid_r_m;return [b.l+(z-zs[0])/(zs.at(-1)-zs[0])*(b.r-b.l),b.b-(r-rs[0])/(rs.at(-1)-rs[0])*(b.b-b.t)]}
function drawOverlay(ctx,c,b){const B=c.body,P=(z,r)=>mapPoint(z,r,c,b);ctx.save();
// thruster body: hatched region outside the channel wall for z < z_exit
const poly=[...B.wall_profile_zr_m.map(([z,r])=>P(z,r)),P(B.z_exit_m,B.r_exit_m),P(B.z_exit_m,B.r_plume_m),P(c.grid_z_m[0],B.r_plume_m)];
ctx.beginPath();poly.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle="#7f9a9333";ctx.fill();ctx.clip();ctx.strokeStyle="#7f9a93aa";ctx.lineWidth=1;for(let x=b.l-b.b;x<b.r+b.b;x+=10){ctx.beginPath();ctx.moveTo(x,b.b);ctx.lineTo(x+(b.b-b.t),b.t);ctx.stroke()}ctx.restore();ctx.save();
ctx.lineWidth=2;ctx.strokeStyle="#9bb8b0";ctx.beginPath();B.wall_profile_zr_m.forEach(([z,r],i)=>{const p=P(z,r);i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])});ctx.stroke();
const seg=(z0,r0,z1,r1,col,dash)=>{ctx.beginPath();ctx.setLineDash(dash||[]);ctx.strokeStyle=col;const a=P(z0,r0),q=P(z1,r1);ctx.moveTo(a[0],a[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.setLineDash([])};
seg(c.grid_z_m[0],0,c.grid_z_m[0],B.r_bore_m,"#ff6b6b");seg(B.z_exit_m,B.r_exit_m,B.z_exit_m,B.r_body_dielectric_m,"#ffcf67");seg(B.z_exit_m,B.r_body_dielectric_m,B.z_exit_m,B.r_plume_m,"#58a8ff");
ctx.lineWidth=1.2;seg(B.z_exit_m,B.r_plume_m,B.z_max_m,B.r_plume_m,"#eef7f4",[2,3]);seg(B.z_max_m,0,B.z_max_m,B.r_plume_m,"#eef7f4",[2,3]);
const ca=c.config.cathode;if(ca){const a=P(ca.z_start_m,ca.r_outer_m),q=P(ca.z_end_m,ca.r_inner_m);ctx.setLineDash([4,3]);ctx.strokeStyle="#c58bff";ctx.strokeRect(a[0],a[1],q[0]-a[0],q[1]-a[1]);ctx.setLineDash([])}
ctx.setLineDash([5,4]);ctx.strokeStyle="#ffffffaa";ctx.lineWidth=1;c.cusps.cusp_z_m.forEach(zc=>{const p=P(zc,0);ctx.beginPath();ctx.moveTo(p[0],P(0,B.r_exit_m)[1]);ctx.lineTo(p[0],b.b);ctx.stroke()});ctx.setLineDash([]);
if(showTracks){ctx.lineWidth=1;c.trajectories.tracks.forEach((t,k)=>{ctx.strokeStyle=k%2?"#ffffffcc":"#ffe9a8cc";ctx.beginPath();t.zr_m.forEach(([z,r],i)=>{const p=P(z,r);i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])});ctx.stroke();const s=P(t.zr_m[0][0],t.zr_m[0][1]);ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(s[0],s[1],2.2,0,2*Math.PI);ctx.fill()})}
ctx.restore()}
function drawField(){const c=DATA.cases[selected],s=setup($("field")),ctx=s.c,b=bounds(s.w,s.h),view=viewMatrix(c,mapKey),range=viewRange(view,mapKey);
ctx.clearRect(0,0,s.w,s.h);ctx.fillStyle=themeColor("--panel");ctx.fillRect(0,0,s.w,s.h);paintView(ctx,b,view,range);drawOverlay(ctx,c,b);
axes(ctx,b,s.w,s.h,"z (m)","r (m)",c.grid_z_m[0],c.grid_z_m.at(-1),c.grid_r_m[0],c.grid_r_m.at(-1));drawColorbar(ctx,s,b,range);mapCaption(c,view,mapKey);
if(cursor){const p=mapPoint(cursor.z,cursor.r,c,b);ctx.strokeStyle="#fff";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(p[0]-8,p[1]);ctx.lineTo(p[0]+8,p[1]);ctx.moveTo(p[0],p[1]-8);ctx.lineTo(p[0],p[1]+8);ctx.stroke()}}
function tick(v,lo,hi){const m=Math.max(Math.abs(lo),Math.abs(hi));return m>=1e5||(m>0&&m<1e-2)?(v===0?"0":Number(v).toExponential(2)):fmt(v,3)}
function axes(c,b,w,h,xlabel,ylabel,xmin,xmax,ymin,ymax,ylog=false){c.strokeStyle=themeColor("--line");c.fillStyle=themeColor("--muted");c.lineWidth=1;c.font="12px system-ui";c.strokeRect(b.l,b.t,b.r-b.l,b.b-b.t);c.textAlign="center";for(let i=0;i<=4;i++){const x=b.l+(b.r-b.l)*i/4;c.fillText(tick(xmin+(xmax-xmin)*i/4,xmin,xmax),x,b.b+18)}c.fillText(xlabel,(b.l+b.r)/2,h-6);c.save();c.translate(13,(b.t+b.b)/2);c.rotate(-Math.PI/2);c.fillText(ylabel,0,0);c.restore();c.textAlign="right";for(let i=0;i<=4;i++){const v=ymax-(ymax-ymin)*i/4;c.fillText(ylog?"1e"+fmt(v,3):tick(v,ymin,ymax),b.l-6,b.t+(b.b-b.t)*i/4+4)}c.textAlign="left"}
function quantile(values,q){const s=[...values].sort((a,b)=>a-b);if(!s.length)return NaN;const k=(s.length-1)*q,i=Math.floor(k);return s[i]+(s[Math.min(i+1,s.length-1)]-s[i])*(k-i)}
function updateCursor(clientX,clientY){const canvas=$("field"),rect=canvas.getBoundingClientRect(),b=bounds(rect.width,rect.height),c=DATA.cases[selected],x=Math.max(b.l,Math.min(b.r,clientX-rect.left)),y=Math.max(b.t,Math.min(b.b,clientY-rect.top));cursor=viewCursor(c,viewMatrix(c,mapKey),b,x,y);showTip(clientX-rect.left,clientY-rect.top);schedule(false)}
function showTip(x,y){const c=DATA.cases[selected],t=$("tip");if(!cursor){t.style.display="none";return}t.textContent=cellText(c,viewMatrix(c,mapKey),mapKey);t.style.display="block";t.style.left=Math.min(x+12,t.parentElement.clientWidth-t.offsetWidth-5)+"px";t.style.top=Math.max(4,y-36)+"px"}
function drawPlot(id,series,xLabel,yLabel,log=false,marks={}){const s=setup($(id)),c=s.c,b={l:64,t:16,r:s.w-16,b:s.h-40},pts=series.filter(q=>q.x.length);if(!pts.length){c.clearRect(0,0,s.w,s.h);return}const all=pts.flatMap(q=>q.y.filter(v=>v!=null&&isFinite(v)&&(!log||v>0)));if(!all.length){c.clearRect(0,0,s.w,s.h);return}const xmin=marks.xmin!=null?marks.xmin:Math.min(...pts.flatMap(q=>q.x)),xmax=marks.xmax!=null?marks.xmax:Math.max(...pts.flatMap(q=>q.x));let ymin=Math.min(...all),ymax=Math.max(...all);if(marks.robust){ymin=quantile(all,.002);ymax=quantile(all,.998)}if(marks.ymin!=null)ymin=marks.ymin;if(marks.ymax!=null)ymax=marks.ymax;if(log){ymin=Math.log10(Math.max(ymin,1e-300));ymax=Math.log10(Math.max(ymax,1e-299))}else{const pad=(ymax-ymin||1)*.08;ymin-=pad;ymax+=pad}c.clearRect(0,0,s.w,s.h);c.fillStyle=themeColor("--panel");c.fillRect(0,0,s.w,s.h);const X=x=>b.l+(x-xmin)/(xmax-xmin||1)*(b.r-b.l);
if(marks.band){const x0=Math.max(b.l,X(marks.band[0])),x1=Math.min(b.r,X(marks.band[1]));c.fillStyle=themeColor("--window");c.fillRect(x0,b.t,x1-x0,b.b-b.t)}
if(marks.vlines){c.save();c.setLineDash([4,4]);c.lineWidth=1;marks.vlines.forEach(v=>{const x=typeof v==="number"?v:v.x;if(x==null||x<xmin||x>xmax)return;c.strokeStyle=(v.color)||themeColor("--muted");c.beginPath();c.moveTo(X(x),b.t);c.lineTo(X(x),b.b);c.stroke();if(v.name){c.fillStyle=v.color||themeColor("--muted");c.fillText(v.name,X(x)+3,b.b-6)}});c.restore()}
if(marks.hlines){c.save();c.setLineDash([2,4]);c.lineWidth=1;marks.hlines.forEach(h=>{const yy=log?Math.log10(h.y):h.y;if(!(yy>=ymin&&yy<=ymax))return;c.strokeStyle=h.color||themeColor("--muted");const py=b.b-(yy-ymin)/(ymax-ymin||1)*(b.b-b.t);c.beginPath();c.moveTo(b.l,py);c.lineTo(b.r,py);c.stroke();c.fillStyle=h.color||themeColor("--muted");c.fillText(h.name,b.r-8-c.measureText(h.name).width,py-3)});c.restore()}
axes(c,b,s.w,s.h,xLabel,yLabel,xmin,xmax,ymin,ymax,log);c.save();c.beginPath();c.rect(b.l,b.t,b.r-b.l,b.b-b.t);c.clip();pts.forEach(q=>{c.strokeStyle=q.color;c.lineWidth=q.width||1.6;c.beginPath();let started=false;q.x.forEach((x,i)=>{const v=q.y[i];if(v==null||!isFinite(v)||(log&&v<=0)){started=false;return}const yy=log?Math.log10(v):v,px=X(x),py=b.b-(yy-ymin)/(ymax-ymin||1)*(b.b-b.t);started?c.lineTo(px,py):c.moveTo(px,py);started=true});c.stroke()});c.restore();pts.forEach((q,k)=>{c.fillStyle=q.color;c.fillText(q.name,b.l+8,b.t+14+k*15)})}
function windowOf(c){const r=c.plume&&c.plume.window_step_range,S=c.series;if(!r)return null;const t=S.time_s,st=S.step;const at=k=>{let best=0;for(let i=1;i<st.length;i++)if(Math.abs(st[i]-k)<Math.abs(st[best]-k))best=i;return t[best]*1e6};return [at(r[0]),at(r[1])]}
function drawSeries(){const c=DATA.cases[selected],S=c.series,Q=c.plume_series,t=S.time_s.map(v=>v*1e6),cur=k=>S["current_"+k]||Q["current_"+k]||[],P=DATA.performance,H=c.histograms,B=c.body,band=windowOf(c),tm=band?{band}:{},cz=c.cusps.cusp_z_m.map(v=>v*1e3),ops=c.neutral_inventory||{};
drawPlot("thrust",[{x:t,y:Q.momentum_thrust_total_n.map(v=>v*1e6),name:"T_total (flux + cold gas)",color:"#eef7f4",width:2},{x:t,y:Q.momentum_thrust_flux_n.map(v=>v*1e6),name:"T_flux (far-field momentum flux)",color:"#5ad6c0"},{x:t,y:Q.momentum_thrust_balance_n.map(v=>v*1e6),name:"T_balance = −F_on-thruster (ledger)",color:"#ff6b6b"},{x:t,y:Q.momentum_electrostatic_force_thruster_n.map(v=>-v*1e6),name:"−F_es (Maxwell stress)",color:"#58a8ff"},{x:t,y:Q.momentum_cold_gas_thrust_n.map(v=>v*1e6),name:"cold gas",color:"#ffcf67"}],"t (µs)","µN",false,{...tm,robust:true});
drawPlot("closure",[{x:t,y:Q.momentum_closure_fraction.map(v=>v*100),name:"closure (T_flux − T_balance)/max (%)",color:"#5ad6c0"},{x:t,y:Q.plume_charge_fraction_of_peak.map(v=>v*100),name:"far-field |net charge| / peak n_e (%)",color:"#ff6b6b"}],"t (µs)","%",false,{...tm,robust:true,hlines:[{y:c.config.plume_boundary_gate?c.config.plume_boundary_gate.max_charge_fraction*100:null,name:"gate",color:"#ff6b6b"}].filter(h=>h.y!=null)});
drawPlot("currents",[{x:t,y:cur("discharge_a").map(v=>v*1e3),name:"discharge I_d (anode)",color:"#5ad6c0"},{x:t,y:cur("cathode_emission_a").map(v=>v*1e3),name:"cathode emission",color:"#c58bff"},{x:t,y:cur("exit_ion_beam_a").map(v=>v*1e3),name:"far-field ion current",color:"#ff6b6b"},{x:t,y:cur("exit_electron_a").map(v=>v*1e3),name:"far-field e⁻ current",color:"#ffcf67"},{x:t,y:cur("wall_ion_a").map(v=>v*1e3),name:"wall Xe⁺",color:"#58a8ff"},{x:t,y:cur("body_face_ion_a").map(v=>v*1e3),name:"front-face Xe⁺",color:"#9bb8b0"}],"t (µs)","mA",false,{...tm,robust:true});
const az=c.axis.z_m.map(v=>v*1e3);drawPlot("axis",[{x:az,y:c.axis.phi_v,name:"φ(r = 0, z) (V)",color:"#5ad6c0",width:2},{x:az,y:c.axis.t_e_ev,name:"T_e(r = 0, z) (eV)",color:"#ffcf67"}],"z (mm)","V · eV",false,{vlines:[{x:B.z_exit_m*1e3,name:"exit plane",color:"#9bb8b0"},{x:P.acceleration_z90_m!=null?P.acceleration_z90_m*1e3:null,name:"90 %",color:"#ff6b6b"},{x:P.acceleration_z10_m!=null?P.acceleration_z10_m*1e3:null,name:"10 %",color:"#ff6b6b"},...cz.map(x=>({x}))],hlines:[{y:c.config.potentials.anode_v,name:"U_a",color:"#ff6b6b"}]});
drawPlot("theta",[{x:H.theta_centres_deg,y:H.ion_current_per_sr_a,name:"j_i(θ) (A/sr)",color:"#ff6b6b",width:2},{x:H.theta_centres_deg,y:H.ion_current_cumulative_fraction.map(v=>v*Math.max(...H.ion_current_per_sr_a.filter(x=>x!=null&&isFinite(x)),1e-300)),name:"cumulative fraction (scaled to the peak)",color:"#5ad6c0"}],"θ from the aperture centre (deg)","A/sr",false,{xmin:0,xmax:90,vlines:[{x:P.divergence_half_angle_95_deg,name:"95 % half-angle",color:"#ffcf67"}]});
drawPlot("iedf",[{x:H.iedf_centres_ev,y:H.iedf_counts,name:"macro-ion counts at the far field",color:"#ff6b6b",width:2}],"ion energy (eV)","counts",false,{vlines:[{x:c.config.potentials.anode_v,name:"U_a",color:"#ffcf67"},{x:P.iedf_peak_energy_ev,name:"peak",color:"#5ad6c0"},{x:P.iedf_mean_energy_ev,name:"mean",color:"#58a8ff"}]});
drawPlot("farfield",[{x:c.exit_r_m.map(v=>v*1e3),y:c.exit.exit_ion_current_density_a_per_m2,name:"far plane: ion j_z vs r (mm) (A/m²)",color:"#ff6b6b",width:2},{x:c.exit_r_m.map(v=>v*1e3),y:c.exit.exit_electron_current_density_a_per_m2,name:"far plane: electron j_z vs r (mm)",color:"#5ad6c0"},{x:H.side_z_m.map(v=>v*1e3),y:H.side_ion_current_density_a_per_m2,name:"side cylinder: ion j_r vs z (mm)",color:"#ffcf67"},{x:H.side_z_m.map(v=>v*1e3),y:H.side_electron_current_density_a_per_m2,name:"side cylinder: electron j_r vs z (mm)",color:"#58a8ff"}],"r or z (mm)","A/m²");
drawPlot("counts",[{x:t,y:S.electrons,name:"electrons",color:"#5ad6c0"},{x:t,y:S.ions,name:"Xe⁺",color:"#ff6b6b"}],"t (µs)","macro-particles",false,tm);
drawPlot("neutral",[{x:t,y:S.neutral_fixed_point_per_m3||[],name:"analytic fixed point (noisy)",color:"#ff6b6b",width:1},{x:t,y:S.neutral_density_per_m3||[],name:"n_g,channel(t)",color:"#5ad6c0",width:1.4}],"t (µs)","n_g (m⁻³)",false,{...tm,robust:true,hlines:[{y:ops.zero_ionization_density_per_m3,name:"n_g0 = Q_in/c",color:"#ffcf67"}].filter(h=>h.y!=null)});
drawPlot("energy",[{x:t,y:S.total_energy_j,name:"K+U total",color:"#eef7f4"},{x:t,y:S.kinetic_electron_j,name:"K electrons",color:"#5ad6c0"},{x:t,y:S.kinetic_ion_j,name:"K ions",color:"#ff6b6b"},{x:t,y:S.field_energy_j,name:"U field",color:"#58a8ff"},{x:t,y:(S.interval_electrode_work_j||[]).map(Math.abs),name:"|electrode work| per interval",color:"#c58bff"},{x:t,y:S.interval_residual_j.map(Math.abs),name:"|interval residual|",color:"#ffcf67"}],"t (µs)","energy (J)",true,tm);
drawPlot("wpe",[{x:t,y:S.peak_omega_pe_dt,name:"peak ω_pe Δt",color:"#ffcf67"},{x:t,y:S.peak_node_cells_per_debye||[],name:"peak-node cells / λ_D",color:"#5ad6c0"}],"t (µs)","",false,{...tm,hlines:[{y:.2,name:"ω_pe Δt gate 0.2",color:"#ff6b6b"},{y:c.peak_node_debye&&c.peak_node_debye.gate?c.peak_node_debye.gate.max_cells_per_debye:null,name:"Debye gate",color:"#5ad6c0"}].filter(h=>h.y!=null)});
const wz=c.wall_z_m.map(v=>v*1e3);drawPlot("wall",[{x:wz,y:c.wall.wall_electron_flux_per_m2_s,name:"electron flux (m⁻² s⁻¹)",color:"#5ad6c0"},{x:wz,y:c.wall.wall_ion_flux_per_m2_s,name:"ion flux (m⁻² s⁻¹)",color:"#ff6b6b"}],"z (mm)","flux",true,{vlines:cz,xmax:B.z_exit_m*1e3})}
function drawAll(){renderPerformance();renderDetails();drawField();drawSeries()}
function schedule(full=true){cancelAnimationFrame(raf);raf=requestAnimationFrame(full?drawAll:drawField)}
function select(i){selected=i;caseSelect.value=i;cursor=null;showTip();schedule()}
caseSelect.onchange=()=>select(Number(caseSelect.value));$("map").onchange=e=>{mapKey=e.target.value;schedule()};$("scale").onchange=e=>{scaleMode=e.target.value;schedule(false)};wireMapControls();$("tracks").onchange=e=>{showTracks=e.target.checked;schedule(false)};
$("theme").onclick=()=>{const light=document.documentElement.dataset.theme!=="light";document.documentElement.dataset.theme=light?"light":"dark";$("theme").textContent=light?"Dark theme":"Light theme";$("theme").setAttribute("aria-pressed",light);schedule()};
$("field").addEventListener("pointermove",e=>updateCursor(e.clientX,e.clientY));$("field").addEventListener("pointerleave",()=>{cursor=null;showTip();schedule(false)});
$("field").addEventListener("keydown",e=>{const c=DATA.cases[selected],view=viewMatrix(c,mapKey),zs=view.z,rs=view.r;if(e.key==="Home"){cursor={zi:Math.floor(zs.length/2),ri:0,z:zs[Math.floor(zs.length/2)],r:rs[0]}}else{if(!cursor)cursor={zi:Math.floor(zs.length/2),ri:0,z:zs[Math.floor(zs.length/2)],r:rs[0]};if(e.key==="ArrowLeft")cursor.zi=Math.max(0,cursor.zi-1);else if(e.key==="ArrowRight")cursor.zi=Math.min(zs.length-1,cursor.zi+1);else if(e.key==="ArrowDown")cursor.ri=Math.max(0,cursor.ri-1);else if(e.key==="ArrowUp")cursor.ri=Math.min(rs.length-1,cursor.ri+1);else return;cursor.z=zs[cursor.zi];cursor.r=rs[cursor.ri]}e.preventDefault();showTip(70,30);schedule(false)});
new ResizeObserver(schedule).observe(document.querySelector("main"));window.addEventListener("pageshow",schedule);drawAll();
</script></body></html>
"""


def render_html(payload: Mapping[str, Any]) -> str:
    validate_payload(payload)
    return snapshot_dashboard.fill_template(HTML_TEMPLATE, payload)


def generate(output_path: Path = DEFAULT_OUTPUT, results: Path = RESULTS, protocol_path: Path = PROTOCOL,
             reference_path: Path | None = CHANNEL_REFERENCE) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(build_payload(results, protocol_path, reference_path)), encoding="utf-8", newline="\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-channel-reference", action="store_true", help="omit the channel-only v1.3 reference row")
    args = parser.parse_args()
    print(generate(args.output, args.results, args.protocol, None if args.no_channel_reference else CHANNEL_REFERENCE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
