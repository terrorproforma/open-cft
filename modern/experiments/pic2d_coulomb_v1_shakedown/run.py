"""Box shakedown of model v2.4.0 (``coulomb_v1``) on the ss-v4 33 um protocol - NON-EVIDENTIARY.

``protocol.json`` is the R3 shakedown protocol (the preregistered ``pic2d_cft_steady_state_v4`` protocol with
``operating_point.collision_set = xe_collision_set_v2``) plus ``numerics.coulomb`` and changed identity / status
fields; the R3 shakedown record (``pic2d_xe_collision_set_v2_shakedown/shakedown.json``) is therefore the
Coulomb-OFF twin at the same step count.  Nothing here is a preregistered experiment:

* ``shakedown`` runs the shared steady-state runner for 100 000 steps with the v4 shakedown cadences through
  finalize + assess and records the early Coulomb readings (mean nu_ee / nu_ei / nu_en and nu_ee / nu_en over the
  trailing half of the records, the per-cell frequencies at the peak-density cell and per cusp column from
  ``maps.npz``) next to the R3 twin's S / I_d / T_e,peak in ``shakedown.json``;
* ``cost`` times the Coulomb stage on the device at the protocol's grid with a synthetic plateau-load population
  (default 4.5 M particles: 2.25 M e- + 2.25 M Xe+), stage alone and inside a step (Coulomb cycle on / off), and
  writes ``cost.json`` (ms per launch, ms per step amortised over ``cycle_steps``).

Usage (from the repository's modern/ directory, on the Lambda H100 as an extra MPS client)::

    python -m experiments.pic2d_coulomb_v1_shakedown.run cost --backend warp-cuda
    python -m experiments.pic2d_coulomb_v1_shakedown.run shakedown --backend warp-cuda
    python -m experiments.pic2d_coulomb_v1_shakedown.run readings --results results-shakedown
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.coulomb import column_frequency_profile, coulomb_log_ee
from cft_revival.pic2d.models import XENON_MASS_KG, ParticleArrays
from cft_revival.pic2d.simulation import SimulationState, empty_cumulative
from experiments.pic2d_cft_steady_state_v1 import run as runner
from experiments.pic2d_cft_steady_state_v4 import run as v4

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"
SHAKEDOWN_PATH = HERE / "shakedown.json"
COST_PATH = HERE / "cost.json"
RESULTS = HERE / "results-shakedown"
OFF_TWIN = HERE.parent / "pic2d_xe_collision_set_v2_shakedown" / "shakedown.json"
CUSP_PLANES_M = (0.006028, 0.012, 0.017972)      # P2 cusp planes of the reference design (cusp topology v3.1)
COULOMB_KEYS = ("nu_ee_mean_per_s", "nu_ei_mean_per_s", "nu_en_elastic_mean_per_s", "nu_ee_over_nu_en", "mean_s_ee", "mean_s_ei",
                "fraction_large_s_ee", "fraction_large_s_ei", "mean_coulomb_log_ee", "mean_coulomb_log_ei", "interval_ee_pairs", "interval_ei_pairs")
CURRENT_KEYS = ("ionization_rate_per_s", "discharge_a", "exit_ion_beam_a", "wall_electron_a", "coulomb_nu_ee_mean_per_s", "coulomb_nu_ei_mean_per_s")


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return runner.load_protocol(path)


def _trailing_half_mean(records: list[dict[str, Any]], getter) -> float | None:
    values = []
    for record in records[len(records) // 2:]:
        value = getter(record)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def coulomb_readings(results: Path) -> dict[str, Any]:
    """Early Coulomb readings from the series (trailing half of the records) and the per-cell frequency maps."""

    records = runner._read_jsonl(results / "series.jsonl") if (results / "series.jsonl").is_file() else []
    records = [r for r in records if "currents_a" in r]
    out: dict[str, Any] = {"series_records": len(records)}
    if not records:
        return out
    for key in COULOMB_KEYS:
        out[f"trailing_half_mean_{key}"] = _trailing_half_mean(records, lambda r, k=key: (r.get("coulomb") or {}).get(k))
    for key in CURRENT_KEYS:
        out[f"trailing_half_mean_{key}"] = _trailing_half_mean(records, lambda r, k=key: r["currents_a"].get(k))
    last = records[-1]
    out["last_coulomb_block"] = last.get("coulomb")
    out["cumulative"] = {k: v for k, v in last["ledger"]["cumulative"].items() if k.startswith("coulomb_") or k in ("pz_coulomb", "ke_coulomb_j", "ionizations", "elastic")}
    out["electrons_last"] = last["electrons"]
    out["time_s_last"] = last["time_s"]
    peak = (last.get("peak_node") or {})
    out["peak_node_last"] = {k: peak.get(k) for k in ("node", "n_e_peak_per_m3", "t_e_peak_ev", "cells_per_debye")}
    window = peak.get("window") or {}
    out["peak_window_last"] = {k: window.get(k) for k in ("node", "n_e_peak_per_m3", "t_e_peak_ev", "cells_per_debye")}
    maps_path = results / "maps.npz"
    if maps_path.is_file():
        with np.load(maps_path) as archive:
            maps = {k: np.asarray(archive[k]) for k in archive.files}
        out["maps"] = frequency_map_readings(maps, results)
    return out


def frequency_map_readings(maps: dict[str, np.ndarray], results: Path) -> dict[str, Any]:
    """nu_ee / nu_ei at the window's peak-density cell and the electron-weighted column means at the cusp planes."""

    protocol_path = results / "protocol-shakedown.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else load_protocol()
    config = runner.build_config(protocol, backend="cpu")
    grid = config.grid
    out: dict[str, Any] = {}
    if "coulomb_nu_ee_per_s" not in maps:
        out["note"] = "no Coulomb maps in maps.npz"
        return out
    nu_ee, nu_ei, seconds = maps["coulomb_nu_ee_per_s"], maps["coulomb_nu_ei_per_s"], maps["coulomb_electron_seconds"]
    n_e = maps["n_e_per_m3"]
    # the peak-density CELL: the cell whose four nodes carry the largest mean window density, among cells with electron-seconds
    cells = 0.25 * (n_e[:-1, :-1] + n_e[1:, :-1] + n_e[:-1, 1:] + n_e[1:, 1:])
    weight = seconds[:-1, :-1]
    cells = np.where(weight > 0.0, cells, -1.0)
    i, j = np.unravel_index(int(np.argmax(cells)), cells.shape)
    out["peak_cell"] = {"cell": [int(i), int(j)], "r_m": float((i + 0.5) * grid.dr_m), "z_m": float(grid.geometry.z_min_m + (j + 0.5) * grid.dz_m),
                        "n_e_window_per_m3": float(cells[i, j]), "nu_ee_per_s": float(nu_ee[i, j]), "nu_ei_per_s": float(nu_ei[i, j]),
                        "mean_s_ee": float(maps["coulomb_mean_s_ee"][i, j]), "electron_seconds": float(weight[i, j])}
    out["cusp_columns_nu_ee_per_s"] = column_frequency_profile(nu_ee, seconds, grid, CUSP_PLANES_M)
    out["cusp_columns_nu_ei_per_s"] = column_frequency_profile(nu_ei, seconds, grid, CUSP_PLANES_M)
    resolved = weight > 0.0
    out["electron_weighted_mean_nu_ee_per_s"] = float(np.sum(nu_ee[:-1, :-1][resolved] * weight[resolved]) / weight[resolved].sum()) if resolved.any() else None
    out["electron_weighted_mean_nu_ei_per_s"] = float(np.sum(nu_ei[:-1, :-1][resolved] * weight[resolved]) / weight[resolved].sum()) if resolved.any() else None
    out["max_nu_ee_per_s"] = float(np.max(nu_ee))
    out["definition_note"] = ("coulomb_nu_*_per_s are the operator's pair-mean deflection rates <s> / dt_c (a 1/g^3-weighted mean: heavy-tailed, "
                              "several times the thermal rate); nu_e_spitzer_* is the NRL electron collision rate 2.91e-6 n lnL T^-3/2 from the window "
                              "n_e / T_e maps (the audit's definition)")
    if "t_e_ev" in maps:
        t_nodes = maps["t_e_ev"]
        t_cells = 0.25 * (t_nodes[:-1, :-1] + t_nodes[1:, :-1] + t_nodes[:-1, 1:] + t_nodes[1:, 1:])
        n_cells = 0.25 * (n_e[:-1, :-1] + n_e[1:, :-1] + n_e[:-1, 1:] + n_e[1:, 1:])
        out["peak_cell"]["t_e_window_ev"] = float(t_cells[i, j])
        floor = float(protocol["numerics"].get("coulomb", {}).get("coulomb_log_floor", 2.0))
        valid = (n_cells > 0.0) & (t_cells > 0.0)
        lnl = np.where(valid, coulomb_log_ee(np.where(valid, n_cells, 1.0), np.where(valid, t_cells, 1.0), floor), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            spitzer = np.where(valid, 2.91e-6 * n_cells * 1e-6 * lnl * np.where(valid, t_cells, 1.0) ** -1.5, 0.0)
        spitzer_nodes = np.zeros(nu_ee.shape)
        spitzer_nodes[:-1, :-1] = spitzer
        out["peak_cell"]["nu_e_spitzer_per_s"] = float(spitzer[i, j])
        out["peak_cell"]["coulomb_log_ee"] = float(lnl[i, j])
        out["cusp_columns_nu_e_spitzer_per_s"] = column_frequency_profile(spitzer_nodes, seconds, grid, CUSP_PLANES_M)
        out["electron_weighted_mean_nu_e_spitzer_per_s"] = float(np.sum(spitzer[resolved] * weight[resolved]) / weight[resolved].sum()) if resolved.any() else None
    return out


def off_twin_readings(path: Path = OFF_TWIN) -> dict[str, Any] | None:
    """The R3 (Coulomb-off) shakedown record's S / I_d / T_e,peak readings at the same 100k steps."""

    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    readings = record.get("collision_readings") or {}
    window = (record.get("peak_debye_window") or {}).get("last") or {}
    return {
        "record": str(path.relative_to(HERE.parent.parent)), "git_head": record.get("git_head"), "steps_completed": record.get("steps_completed"),
        "trailing_half_mean_ionization_rate_per_s": readings.get("trailing_half_mean_ionization_rate_per_s"),
        "trailing_half_mean_discharge_a": readings.get("trailing_half_mean_discharge_a"),
        "trailing_half_mean_exit_ion_beam_a": readings.get("trailing_half_mean_exit_ion_beam_a"),
        "peak_window_last": {k: window.get(k) for k in ("node", "n_e_peak_per_m3", "t_e_peak_ev", "cells_per_debye")},
        "final_counts": record.get("final_counts"), "ms_per_step": record.get("ms_per_step"),
    }


def direction_against_off(on: dict[str, Any], off: dict[str, Any] | None) -> dict[str, Any]:
    if off is None:
        return {"note": "off twin record not found"}
    out: dict[str, Any] = {}
    for key_on, key_off, label in (
        ("trailing_half_mean_ionization_rate_per_s", "trailing_half_mean_ionization_rate_per_s", "S"),
        ("trailing_half_mean_discharge_a", "trailing_half_mean_discharge_a", "I_d"),
        ("trailing_half_mean_exit_ion_beam_a", "trailing_half_mean_exit_ion_beam_a", "I_beam"),
    ):
        a, b = on.get(key_on), off.get(key_off)
        out[label] = {"coulomb_on": a, "coulomb_off": b, "relative_change": (a - b) / b if a is not None and b not in (None, 0.0) else None}
    t_on = (on.get("peak_window_last") or {}).get("t_e_peak_ev")
    t_off = (off.get("peak_window_last") or {}).get("t_e_peak_ev")
    out["T_e_peak_window"] = {"coulomb_on": t_on, "coulomb_off": t_off, "relative_change": (t_on - t_off) / t_off if t_on and t_off else None}
    n_on = (on.get("peak_window_last") or {}).get("n_e_peak_per_m3")
    n_off = (off.get("peak_window_last") or {}).get("n_e_peak_per_m3")
    out["n_e_peak_window"] = {"coulomb_on": n_on, "coulomb_off": n_off, "relative_change": (n_on - n_off) / n_off if n_on and n_off else None}
    out["note"] = ("100k-step (0.14 us) seed-transient readings: a direction indicator only; the audit's expectations (S +5-20 %, T_e,peak -5 %) "
                   "are plateau statements")
    return out


def _attach_readings(record: dict[str, Any], protocol: dict[str, Any], results: Path) -> dict[str, Any]:
    record["schema_version"] = "cft-revival.pic2d-coulomb-v1.shakedown/1.0.0"
    record["coulomb"] = protocol["numerics"]["coulomb"]
    record["coulomb_readings"] = coulomb_readings(results)
    record["off_twin"] = off_twin_readings()
    record["direction_against_off_twin"] = direction_against_off(record["coulomb_readings"], record["off_twin"])
    record["not_a_result"] = "100k-step shakedown of model v2.4.0 on the ss-v4 protocol; early readings only"
    return record


def shakedown(backend: str = "warp-cuda") -> dict[str, Any]:
    protocol = load_protocol()
    record = v4.shakedown(protocol, results=RESULTS, backend=backend, output=SHAKEDOWN_PATH)
    _attach_readings(record, protocol, RESULTS)
    artifacts.write_canonical_json(SHAKEDOWN_PATH, record)
    print(json.dumps({"coulomb_readings": record["coulomb_readings"], "direction": record["direction_against_off_twin"]}, indent=1, default=str), flush=True)
    return record


def refresh_readings(results: Path = RESULTS, output: Path = SHAKEDOWN_PATH) -> dict[str, Any]:
    """Recompute the readings blocks of an existing shakedown record from its results directory (CPU only; the run record -
    steps, timing, gates, assessment - is kept as written by the run)."""

    record = json.loads(output.read_text(encoding="utf-8"))
    record["readings_refreshed_git_head"] = runner.git_head()
    _attach_readings(record, load_protocol(), results)
    artifacts.write_canonical_json(output, record)
    print(json.dumps({"coulomb_readings": record["coulomb_readings"], "direction": record["direction_against_off_twin"]}, indent=1, default=str), flush=True)
    return record


# -- cost probe --------------------------------------------------------------------------------------------------------------------

def cost(backend: str = "warp-cuda", *, particles: int = 4_500_000, repeats: int = 20, output: Path = COST_PATH) -> dict[str, Any]:
    """Time the Coulomb stage on the protocol's grid with a synthetic plateau-load population (uniform in the channel volume,
    Maxwellian electrons at 6 eV, cold ions), stage alone (launch + device sync) and inside the captured step (cycle on / off)."""

    import warp as wp

    from cft_revival.pic2d.fields import uniform_field_map
    from cft_revival.pic2d.simulation import Simulation

    protocol = load_protocol()
    config = runner.build_config(protocol, backend=backend)
    xs = config.mcc.collision_set.load_electron_cross_sections() if config.mcc is not None and config.mcc.collision_set is not None else None
    grid = config.grid
    field = uniform_field_map(grid, 0.05)
    rng = np.random.default_rng(1)
    n_e = particles // 2
    volume_fraction = np.sqrt(rng.random(n_e))
    r = grid.geometry.bore_radius_m * volume_fraction * (1.0 - 1e-9)
    z = grid.geometry.z_min_m + grid.geometry.channel_length_m * rng.random(n_e) * (1.0 - 1e-9)
    from cft_revival.pic2d.models import ELECTRON_MASS_KG, EV_J

    s_e = np.sqrt(6.0 * EV_J / ELECTRON_MASS_KG)
    electrons = ParticleArrays(r, z, rng.normal(0.0, s_e, n_e), rng.normal(0.0, s_e, n_e), rng.normal(0.0, s_e, n_e))
    s_i = np.sqrt(0.1 * EV_J / XENON_MASS_KG)
    ri = grid.geometry.bore_radius_m * np.sqrt(rng.random(n_e)) * (1.0 - 1e-9)
    zi = grid.geometry.z_min_m + grid.geometry.channel_length_m * rng.random(n_e) * (1.0 - 1e-9)
    ions = ParticleArrays(ri, zi, rng.normal(0.0, s_i, n_e), rng.normal(0.0, s_i, n_e), rng.normal(0.0, s_i, n_e))
    device = "cuda:0" if backend == "warp-cuda" else "cpu"
    sim = Simulation(config, field, cross_sections=xs, backend=backend, step_graph=(backend == "warp-cuda"))
    # the seeded state with its neutral inventory, the particles replaced by the synthetic plateau load
    state: SimulationState = sim.state
    state.electrons = electrons
    state.ions = ions
    state.cumulative = empty_cumulative()
    sim.load_state(state)
    backend_obj = sim.backend
    dev = backend_obj.device
    out: dict[str, Any] = {"backend": backend, "device": str(dev), "particles": {"electrons": n_e, "ions": n_e}, "grid": grid.to_dict(),
                           "cycle_steps": config.coulomb.cycle_steps, "gpu": None}
    try:
        out["gpu"] = wp.get_device(device).name
    except (RuntimeError, ValueError, KeyError):  # pragma: no cover - an unnamed device keeps None
        out["gpu"] = None
    stage = backend_obj.coulomb
    e_species, i_species = backend_obj.species["e"], backend_obj.species["i"]

    def timed(fn, n: int) -> list[float]:
        times = []
        for _ in range(n):
            wp.synchronize_device(dev)
            t0 = time.perf_counter()
            fn()
            wp.synchronize_device(dev)
            times.append((time.perf_counter() - t0) * 1e3)
        backend_obj.step_counter.zero_()      # keep the seed-table row inside the uploaded sync interval
        return times

    # 1. the stage alone (direct launches): full cycle, and the sort / pairing split
    dt_c = config.dt_s * config.coulomb.cycle_steps
    full = lambda: stage.launch(e_species, i_species, backend_obj.slots, backend_obj.seed_table, backend_obj.step_counter, e_species.capacity, i_species.capacity, dt_c, False)
    timed(full, 3)
    stage_ms = timed(full, repeats)
    sort_only = lambda: (
        stage._sort(e_species, 0, e_species.capacity, stage.counts_e, stage.starts_e, stage.cell_e, stage.pos_e, stage.tmp_e, stage.sorted_e, stage.cell_of_sorted_e, backend_obj.slots),
        stage._sort(i_species, 1, i_species.capacity, stage.counts_i, stage.starts_i, stage.cell_i, stage.pos_i, stage.tmp_i, stage.sorted_i, stage.cell_of_sorted_i, backend_obj.slots),
    )
    sort_ms = timed(sort_only, repeats)
    out["stage_alone_ms"] = {"full_cycle_median": float(np.median(stage_ms)), "full_cycle_min": float(np.min(stage_ms)),
                             "cell_sort_both_species_median": float(np.median(sort_ms)),
                             "pairing_and_collisions_median": float(np.median(stage_ms) - np.median(sort_ms)),
                             "capacities": {"electrons": e_species.capacity, "ions": i_species.capacity}}
    # 2. inside the step: a Coulomb-cycle step vs an ordinary step (both through the captured graph when available)
    sim.load_state(state)
    step_on = lambda: backend_obj._step_graph_launch(True, True, False, False, True) if backend_obj.step_graph else backend_obj._launch_step(True, True, False, fixed_shape=False, coulomb_step=True)
    step_off = lambda: backend_obj._step_graph_launch(True, True, False, False, False) if backend_obj.step_graph else backend_obj._launch_step(True, True, False, fixed_shape=False, coulomb_step=False)
    timed(step_on, 2)
    timed(step_off, 2)
    on_ms = timed(step_on, repeats)
    off_ms = timed(step_off, repeats)
    k = config.coulomb.cycle_steps
    out["step_ms"] = {"coulomb_cycle_step_median": float(np.median(on_ms)), "ordinary_step_median": float(np.median(off_ms)),
                      "coulomb_overhead_per_cycle_ms": float(np.median(on_ms) - np.median(off_ms)),
                      "amortised_overhead_per_step_ms": float((np.median(on_ms) - np.median(off_ms)) / k),
                      "amortised_overhead_fraction": float((np.median(on_ms) - np.median(off_ms)) / k / np.median(off_ms)),
                      "graph": bool(backend_obj.step_graph_active), "note": "ion step + ion redeposit variant; no window accumulation; "
                      "the step counter advanced, so the seed rows differ between repeats (statistics only)"}
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["git_head"] = runner.git_head()
    artifacts.write_canonical_json(output, out)
    print(json.dumps({k_: v for k_, v in out.items() if k_ in ("stage_alone_ms", "step_ms", "gpu", "particles")}, indent=1), flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shakedown", help="100k-step non-evidentiary shakedown through finalize + assess")
    s.add_argument("--backend", default="warp-cuda")
    c = sub.add_parser("cost", help="time the Coulomb stage at the plateau load on the protocol's grid")
    c.add_argument("--backend", default="warp-cuda")
    c.add_argument("--particles", type=int, default=4_500_000)
    c.add_argument("--repeats", type=int, default=20)
    r = sub.add_parser("readings", help="print the Coulomb readings of the shakedown results")
    r.add_argument("--results", type=Path, default=RESULTS)
    f = sub.add_parser("refresh-readings", help="recompute the readings blocks of shakedown.json from the results directory (CPU only)")
    f.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    if args.command == "shakedown":
        shakedown(backend=args.backend)
    elif args.command == "cost":
        cost(backend=args.backend, particles=args.particles, repeats=args.repeats)
    elif args.command == "refresh-readings":
        refresh_readings(results=args.results)
    else:
        print(json.dumps(coulomb_readings(args.results), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
