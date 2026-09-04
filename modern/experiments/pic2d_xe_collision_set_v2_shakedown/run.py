"""Box shakedown of model v2.3.0 (``xe_collision_set_v2``) on the ss-v4 33 um protocol - NON-EVIDENTIARY.

``protocol.json`` is the preregistered ``pic2d_cft_steady_state_v4`` protocol with ``operating_point.collision_set``
added (four Biagi-v7.1 excitation levels + Xe+ / Xe CEX and MEX) and its identity / status fields changed.  Nothing
here is a preregistered experiment: ``shakedown`` runs the shared steady-state runner for 100 000 steps with the v4
shakedown cadences through finalize + assess and records the early collision readings the R3 brief asks for
(CEX / MEX event rates, fast-neutral bookkeeping, the exit-plane IEDF shape) in ``shakedown.json``; ``compare-iedf``
puts the shakedown's IEDF next to the legacy plateau IEDF of a recorded run.

Usage (from the repository's modern/ directory, on the Lambda H100 as an extra MPS client)::

    python -m experiments.pic2d_xe_collision_set_v2_shakedown.run shakedown --backend warp-cuda
    python -m experiments.pic2d_xe_collision_set_v2_shakedown.run compare-iedf --reference ../path/to/ss-v4/results
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
COLLISION_CURRENT_KEYS = (
    "cex_rate_per_s", "mex_rate_per_s", "cex_plume_rate_per_s", "fast_neutral_exit_rate_per_s", "fast_neutral_wall_rate_per_s",
    "fast_neutral_thermal_rate_per_s", "ion_mcc_candidate_rate_per_s", "ionization_rate_per_s", "discharge_a", "exit_ion_beam_a",
)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return runner.load_protocol(path)


def iedf_shape(iedf: np.ndarray, edges_ev: np.ndarray, anode_v: float = 300.0) -> dict[str, Any]:
    """Normalised exit-plane IEDF descriptors: low-energy population fractions, mean / median energy, peak bin."""

    counts = np.asarray(iedf, dtype=np.float64)
    total = float(counts.sum())
    edges = np.asarray(edges_ev, dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    if total <= 0.0:
        return {"total_macro_ions": 0.0}
    pdf = counts / total
    cdf = np.cumsum(pdf)
    anode = float(anode_v)
    iedf_max_ev = float(edges[-1])
    return {
        "total_macro_ions": total,
        "mean_energy_ev": float(np.sum(pdf * centres)),
        "median_energy_ev": float(centres[int(np.searchsorted(cdf, 0.5))]),
        "peak_energy_ev": float(centres[int(np.argmax(counts))]),
        "fraction_below_10pct_anode": float(cdf[int(np.searchsorted(centres, 0.1 * anode))]),
        "fraction_below_25pct_anode": float(cdf[int(np.searchsorted(centres, 0.25 * anode))]),
        "fraction_below_50pct_anode": float(cdf[int(np.searchsorted(centres, 0.5 * anode))]),
        "fraction_above_90pct_anode": float(1.0 - cdf[int(np.searchsorted(centres, 0.9 * anode))]),
        "bins": int(counts.size), "iedf_max_ev": float(iedf_max_ev),
    }


def collision_readings(results: Path) -> dict[str, Any]:
    """Early CEX / MEX readings from the series (trailing half of the records) and the IEDF shape from maps.npz."""

    records = runner._read_jsonl(results / "series.jsonl") if (results / "series.jsonl").is_file() else []
    records = [r for r in records if "currents_a" in r]
    out: dict[str, Any] = {"series_records": len(records)}
    if records:
        tail = records[len(records) // 2:]
        for key in COLLISION_CURRENT_KEYS:
            values = [float(r["currents_a"].get(key, float("nan"))) for r in tail]
            if np.all(np.isfinite(values)):
                out[f"trailing_half_mean_{key}"] = float(np.mean(values))
        last = records[-1]["ledger"]["cumulative"]
        out["cumulative"] = {k: last.get(k) for k in ("cex", "mex", "cex_plume", "ion_mcc_candidates", "ion_mcc_null", "fast_neutral_exit_channel",
                                                       "fast_neutral_wall", "fast_neutral_thermal", "fast_neutral_unresolved", "ion_mcc_ceiling_violations",
                                                       "ion_neutral_loss_j", "pz_ion_collisions", "pz_fast_neutral_exit", "pz_fast_neutral_wall",
                                                       "ke_fast_neutral_exit_j", "excitations", "excitations_level_1", "excitations_level_2",
                                                       "excitations_level_3", "excitations_level_4", "ionizations", "inelastic_loss_j", "exit_ions")}
        cex, exit_ions = float(last.get("cex", 0.0)), float(last.get("exit_ions", 0.0))
        out["cex_events_per_exit_ion"] = cex / exit_ions if exit_ions > 0 else None
        out["ions_last"] = records[-1]["ions"]
        out["time_s_last"] = records[-1]["time_s"]
        if records[-1].get("neutral") is not None:
            out["neutral_last"] = {k: records[-1]["neutral"].get(k) for k in ("density_per_m3", "fixed_point_per_m3", "fast_neutral_exit_rate_per_s", "ionization_rate_per_s", "effusion_rate_per_s")}
    shape = iedf_from_maps(results)
    if shape is not None:
        out["iedf_exit_plane"] = shape
    return out


def iedf_from_maps(results: Path) -> dict[str, Any] | None:
    maps_path = results / "maps.npz"
    if not maps_path.is_file():
        return None
    maps = np.load(maps_path)
    if "iedf_ion_counts" not in maps or "iedf_edges_ev" not in maps:
        return None
    anode = 300.0
    protocol_path = results / "protocol-shakedown.json"
    if protocol_path.is_file():
        anode = float(json.loads(protocol_path.read_text(encoding="utf-8"))["operating_point"]["anode_potential_v"])
    return iedf_shape(maps["iedf_ion_counts"], maps["iedf_edges_ev"], anode)


def shakedown(backend: str = "warp-cuda") -> dict[str, Any]:
    protocol = load_protocol()
    record = v4.shakedown(protocol, results=RESULTS, backend=backend, output=SHAKEDOWN_PATH)
    record["schema_version"] = "cft-revival.pic2d-xe-collision-set-v2.shakedown/1.0.0"
    record["collision_set"] = protocol["operating_point"]["collision_set"]
    record["collision_readings"] = collision_readings(RESULTS)
    record["not_a_result"] = "100k-step shakedown of model v2.3.0 on the ss-v4 protocol; early readings only"
    artifacts.write_canonical_json(SHAKEDOWN_PATH, record)
    print(json.dumps(record["collision_readings"], indent=1, default=str), flush=True)
    return record


def compare_iedf(reference_results: Path, output: Path | None = None) -> dict[str, Any]:
    """The shakedown's exit-plane IEDF shape next to a recorded legacy run's (a shape reference, not a like-for-like)."""

    out: dict[str, Any] = {"shakedown": iedf_from_maps(RESULTS), "reference_results": str(reference_results), "reference": iedf_from_maps(reference_results)}
    target = HERE / "iedf-comparison.json" if output is None else output
    artifacts.write_canonical_json(target, out)
    print(json.dumps(out, indent=1, default=str), flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shakedown", help="100k-step non-evidentiary shakedown through finalize + assess")
    s.add_argument("--backend", default="warp-cuda")
    c = sub.add_parser("compare-iedf", help="IEDF shape of the shakedown next to a recorded run's maps.npz")
    c.add_argument("--reference", type=Path, required=True)
    r = sub.add_parser("readings", help="print the collision readings of the shakedown results")
    r.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    if args.command == "shakedown":
        shakedown(backend=args.backend)
    elif args.command == "compare-iedf":
        compare_iedf(args.reference)
    else:
        print(json.dumps(collision_readings(args.results), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
