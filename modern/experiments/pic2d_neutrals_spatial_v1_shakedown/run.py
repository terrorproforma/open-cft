"""Box shakedown of model v2.5.0 (``neutrals_spatial_v1`` + ``metastables_v1``) on the ss-v4 33 um protocol - NON-EVIDENTIARY.

``protocol.json`` is the R3 shakedown protocol (the preregistered ``pic2d_cft_steady_state_v4`` protocol with
``operating_point.collision_set``) with ``operating_point.neutral_inventory`` REPLACED by ``operating_point.neutrals``
(the spatial model with metastables) and the MCC ceiling raised to the Knudsen anode density with headroom.  Nothing here
is a preregistered experiment: ``shakedown`` runs the shared steady-state runner for 100 000 steps with the v4 shakedown
cadences through finalize + assess and records the early neutral readings the R5 brief asks for in ``shakedown.json``:
the axis density profile (window-mean map), where the ionisation sits (axial profile of the ionisation-rate map against
the 0-D shakedown's), the metastable fraction, the atom-ledger identities and the sub-step cost.

Usage (from the repository's modern/ directory, on the Lambda H100 as an extra MPS client)::

    python -m experiments.pic2d_neutrals_spatial_v1_shakedown.run shakedown --backend warp-cuda
    python -m experiments.pic2d_neutrals_spatial_v1_shakedown.run readings --results results-shakedown --reference ../pic2d_xe_collision_set_v2_shakedown/results-shakedown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.pic2d import artifacts
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v4 import run as v4

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
SHAKEDOWN_PATH = HERE / "shakedown.json"
RESULTS = HERE / "results-shakedown"
NEUTRAL_KEYS = (
    "density_per_m3", "axis_density_anode_per_m3", "axis_density_exit_per_m3", "density_max_per_m3", "atoms_ground", "atoms_metastable",
    "macro_neutrals", "macro_metastables", "ionization_rate_per_s", "effusion_rate_per_s", "recycled_rate_per_s", "fast_neutral_in_rate_per_s",
    "cex_converted_rate_per_s", "gross_utilisation", "net_utilisation", "neutral_exit_thrust_n", "debt_ground_atoms", "pending_atoms",
    "interval_ledger_residual_atoms", "interval_meta_ledger_residual_atoms", "sink_consistency_atoms", "neutral_time_s", "substeps",
    "ceiling_violation_fraction",
)
META_KEYS = ("channel_mean_density_per_m3", "fraction_of_ground", "production_rate_per_s", "stepwise_ionization_rate_per_s", "superelastic_rate_per_s",
             "wall_deexcitation_rate_per_s", "stepwise_fraction_of_ionization")


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return runner.load_protocol(path)


def axial_profile(cell_map: np.ndarray, radial_cells: int | None = None) -> list[float]:
    """Volume-weighted axial profile of a cell-centred map over the inner cells (or the axis column when ``radial_cells`` is 1)."""

    values = np.asarray(cell_map, dtype=np.float64)
    inner = values if radial_cells is None else values[:radial_cells]
    return [float(v) for v in np.mean(inner, axis=0)]


def ionization_location(ionization_map: np.ndarray, z_m: np.ndarray) -> dict[str, Any]:
    """Where the ionisation sits: axial profile of the node ionisation-rate map, its centroid and quartiles along z."""

    node_map = np.asarray(ionization_map, dtype=np.float64)
    profile = np.sum(np.nan_to_num(node_map), axis=0)
    total = float(profile.sum())
    if total <= 0.0:
        return {"total": 0.0}
    cdf = np.cumsum(profile) / total
    z = np.asarray(z_m, dtype=np.float64)
    quartile = lambda q: float(z[min(int(np.searchsorted(cdf, q)), z.size - 1)])  # noqa: E731
    return {
        "total": total,
        "centroid_z_m": float(np.sum(profile * z) / total),
        "z25_m": quartile(0.25), "z50_m": quartile(0.5), "z75_m": quartile(0.75),
        "fraction_upstream_of_12mm": float(cdf[min(int(np.searchsorted(z, 0.012)), z.size - 1)]),
        "axial_profile_normalised": [float(v) for v in profile / total],
    }


def neutral_readings(results: Path, reference: Path | None = None) -> dict[str, Any]:
    """Early readings from the series (trailing half) and the window maps; the 0-D shakedown's maps as the comparison, if given."""

    records = runner._read_jsonl(results / "series.jsonl") if (results / "series.jsonl").is_file() else []
    records = [r for r in records if "currents_a" in r and r.get("neutral") is not None]
    out: dict[str, Any] = {"series_records": len(records)}
    if records:
        tail = records[len(records) // 2:]
        out["trailing_half_mean"] = {}
        for key in NEUTRAL_KEYS:
            values = [float(r["neutral"].get(key, float("nan"))) for r in tail]
            if np.all(np.isfinite(values)):
                out["trailing_half_mean"][key] = float(np.mean(values))
        if tail[-1]["neutral"].get("metastables") is not None:
            out["trailing_half_mean_metastables"] = {}
            for key in META_KEYS:
                values = [float(r["neutral"]["metastables"].get(key, float("nan"))) for r in tail]
                if np.all(np.isfinite(values)):
                    out["trailing_half_mean_metastables"][key] = float(np.mean(values))
        last = records[-1]
        out["neutral_last"] = {k: last["neutral"].get(k) for k in NEUTRAL_KEYS}
        out["metastables_last"] = last["neutral"].get("metastables")
        out["neutral_ledger_cumulative"] = last["neutral"].get("ledger")
        out["max_abs_interval_ledger_residual_atoms"] = max(abs(float(r["neutral"]["interval_ledger_residual_atoms"])) for r in records)
        out["max_abs_interval_meta_ledger_residual_atoms"] = max(abs(float(r["neutral"].get("interval_meta_ledger_residual_atoms", 0.0))) for r in records)
        out["max_abs_sink_consistency_atoms"] = max(abs(float(r["neutral"].get("sink_consistency_atoms", 0.0))) for r in records)
        for key in ("discharge_a", "exit_ion_beam_a", "ionization_rate_per_s", "cex_rate_per_s"):
            values = [float(r["currents_a"].get(key, float("nan"))) for r in tail]
            if np.all(np.isfinite(values)):
                out[f"trailing_half_mean_current_{key}"] = float(np.mean(values))
        c = last["ledger"]["cumulative"]
        out["cumulative"] = {k: c.get(k) for k in ("ionizations", "stepwise_ionizations", "superelastic", "excitations", "cex", "neutral_substeps",
                                                   "neutral_fed", "neutral_ionized", "neutral_effused", "neutral_recycled", "neutral_fast_in",
                                                   "neutral_excited_to_pool", "meta_ionized", "meta_superelastic", "meta_wall_deexcited", "meta_effused")}
        out["time_s_last"] = last["time_s"]
    maps_path = results / "maps.npz"
    if maps_path.is_file():
        maps = np.load(maps_path)
        if "neutral_density_per_m3" in maps:
            density = maps["neutral_density_per_m3"]
            out["axis_neutral_density_profile_per_m3"] = axial_profile(density, 1)
            out["inner_third_neutral_density_profile_per_m3"] = axial_profile(density, max(density.shape[0] // 3, 1))
            out["neutral_density_anode_over_exit_axis"] = float(density[0, 0] / density[0, -1]) if density[0, -1] > 0 else None
            out["neutral_samples"] = int(maps["neutral_samples"][0]) if "neutral_samples" in maps else None
        if "metastable_density_per_m3" in maps and "neutral_density_per_m3" in maps:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(maps["neutral_density_per_m3"] > 0, maps["metastable_density_per_m3"] / np.maximum(maps["neutral_density_per_m3"], 1e-300), 0.0)
            out["axis_metastable_fraction_profile"] = axial_profile(ratio, 1)
            out["metastable_fraction_max"] = float(ratio.max())
        if "ionization_rate_per_m3_s" in maps and "z_m" in maps:
            out["ionization_location"] = ionization_location(maps["ionization_rate_per_m3_s"], maps["z_m"])
        elif "ionization_rate_per_m3_s" in maps:
            nz = maps["ionization_rate_per_m3_s"].shape[1]
            out["ionization_location"] = ionization_location(maps["ionization_rate_per_m3_s"], np.linspace(0.0, 0.024, nz))
    if reference is not None and (reference / "maps.npz").is_file():
        ref = np.load(reference / "maps.npz")
        if "ionization_rate_per_m3_s" in ref:
            nz = ref["ionization_rate_per_m3_s"].shape[1]
            z = ref["z_m"] if "z_m" in ref else np.linspace(0.0, 0.024, nz)
            out["reference_0d_ionization_location"] = ionization_location(ref["ionization_rate_per_m3_s"], z)
            out["reference_results"] = str(reference)
    return out


def shakedown(backend: str = "warp-cuda", reference: Path | None = None) -> dict[str, Any]:
    protocol = load_protocol()
    record = v4.shakedown(protocol, results=RESULTS, backend=backend, output=SHAKEDOWN_PATH)
    record["schema_version"] = "cft-revival.pic2d-neutrals-spatial-v1.shakedown/1.0.0"
    record["neutrals"] = protocol["operating_point"]["neutrals"]
    record["neutral_readings"] = neutral_readings(RESULTS, reference)
    record["not_a_result"] = "100k-step shakedown of model v2.5.0 on the ss-v4 protocol; early readings only (initial Knudsen profile + seed transient)"
    artifacts.write_canonical_json(SHAKEDOWN_PATH, record)
    print(json.dumps({k: v for k, v in record["neutral_readings"].items() if not k.endswith("profile") and not k.endswith("profile_per_m3")},
                     indent=1, default=str), flush=True)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shakedown", help="100k-step non-evidentiary shakedown through finalize + assess")
    s.add_argument("--backend", default="warp-cuda")
    s.add_argument("--reference", type=Path, default=None, help="results dir of the 0-D shakedown (ionisation location comparison)")
    r = sub.add_parser("readings", help="print the neutral readings of the shakedown results")
    r.add_argument("--results", type=Path, default=RESULTS)
    r.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "shakedown":
        shakedown(backend=args.backend, reference=args.reference)
    else:
        print(json.dumps(neutral_readings(args.results, args.reference), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
