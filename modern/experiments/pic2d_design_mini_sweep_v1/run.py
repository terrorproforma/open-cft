"""PIC design mini-sweep v1 - DRAFT runner (no production launch; not preregistered).

From ``modern/`` (``$env:PYTHONPATH="$PWD\\src;$PWD"``)::

    python -m experiments.pic2d_design_mini_sweep_v1.run fields [--design ID]      # CPU: padded P2 solves -> fields/<id>/binding.json (sequential)
    python -m experiments.pic2d_design_mini_sweep_v1.run preflight --domain channel # whole-set preflight -> preflight-channel.json
    python -m experiments.pic2d_design_mini_sweep_v1.run protocol --design ID --domain channel   # print the composed per-design run protocol
    python -m experiments.pic2d_design_mini_sweep_v1.run draft-protocol             # rewrite protocol.json (the DRAFT document)
    python -m experiments.pic2d_design_mini_sweep_v1.run cost                       # cost table + serial schedules
    python -m experiments.pic2d_design_mini_sweep_v1.run run --design ID --domain channel --allow-launch   # REFUSED without the flag
    python -m experiments.pic2d_design_mini_sweep_v1.run status|finalize|targets --design ID --domain channel

``run`` reuses the shared steady-state runner (``experiments.pic2d_cft_steady_state_v1.run``: checkpoints, resume,
plateau rule, gates, frames) with the per-design protocol of ``protocol.build_protocol`` and the design's hash-bound
node field passed in directly; results go to ``results/<design_id>-<domain>/``.  ``targets`` extracts the closure
targets (``closure.extract_targets``) from a finished run's ``maps.npz`` + ``summary.json``.

DRAFT: ``run`` refuses to start unless ``--allow-launch`` is given AND the preflight of the domain option has passed
for every design (the preregistration commit will replace this guard by the sealed protocol).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from . import cost as cost_module
from . import designs as design_module
from . import fields as field_module
from . import preflight as preflight_module
from .closure import extract_targets, load_maps
from .protocol import build_protocol, write_draft_protocol

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"


def results_dir(design_id: str, domain: str) -> Path:
    return RESULTS_ROOT / f"{design_id}-{domain}"


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=1, sort_keys=True, default=str))


def command_fields(args: argparse.Namespace) -> int:
    ids = (args.design,) if args.design else design_module.design_ids()
    for design_id in ids:      # sequential: one host factorisation / solve at a time (BLAS oversubscription lesson)
        binding = field_module.produce_field(design_id)
        print(f"[fields] {design_id}: {'ok' if binding['gates']['all_passed'] else 'GATES FAILED'} -> {field_module.binding_path(design_id)}")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    path, report = preflight_module.write_preflight(args.domain, design_ids=(args.design,) if args.design else None)
    print(f"[preflight] {args.domain}: all_passed={report['all_passed']} over {report['design_count']} designs -> {path}")
    return 0 if report["all_passed"] else 1


def command_protocol(args: argparse.Namespace) -> int:
    target_cell_m, dt_s = GRID_VARIANTS[args.grid]
    protocol, _ = build_protocol(args.design, args.domain, target_cell_m=target_cell_m, dt_s=dt_s)
    _print_json(protocol)
    return 0


def command_draft_protocol(args: argparse.Namespace) -> int:
    summary = None
    path = preflight_module.preflight_path(args.domain)
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = {"domain": report["domain"], "all_passed": report["all_passed"], "design_count": report["design_count"], "generated_utc": report["generated_utc"],
                   "designs": {r["design_id"]: {"passed": r["passed"], "gates": {k: v["passed"] for k, v in r["gates"].items()}} for r in report["designs"]}}
    out = write_draft_protocol(preflight_summary=summary)
    print(f"[draft-protocol] wrote {out}")
    return 0


def command_cost(args: argparse.Namespace) -> int:
    built = [design_module.build_design(d.design_id) for d in design_module.SWEEP_DESIGNS]
    table = cost_module.cost_table(built)
    for domain, rows in table.items():
        print(f"== {domain} ==")
        for row in rows:
            print(f"  {row['design_id']:<28} nodes {row['nodes'][0]}x{row['nodes'][1]:<5} N {row['particles_projected_m']['total_m']:.2f} M  {row['ms_per_step']:.2f} ms/step  transit {row['transit_s']*1e6:.2f} us  "
                  f"{row['steps_to_transits']/1e6:.2f} M steps  {row['wall_hours']:.1f} h  {row['device_gb_projected']:.1f} GB  fact {row['factorisation_s']/60:.1f} min")
    for option in table:
        schedule = cost_module.serial_schedule(table, option=option)
        print(f"serial schedule {option}: {schedule['total_hours']:.1f} h = {schedule['total_days_at_24h']:.2f} days for {len(schedule['items'])} runs")
    print("anchors:", json.dumps(cost_module.anchor_residuals()))
    return 0


def _runner():
    from experiments.pic2d_cft_steady_state_v1 import run as runner

    return runner


GRID_VARIANTS = {"50um": (None, None), "33um": (cost_module.REFINED_CHANNEL_CELL_M, cost_module.REFINED_CHANNEL_DT_S)}


def _prepare(design_id: str, domain: str, grid: str = "50um"):
    """Field first, then the protocol (the dt rule and the cathode placement read the design's own node field)."""

    target_cell_m, dt_s = GRID_VARIANTS[grid]
    built = design_module.build_design(design_id)
    mapping = design_module.pic_geometry(built, domain) if target_cell_m is None else design_module.pic_geometry(built, domain, target_cell_m=target_cell_m)
    binding = field_module.load_binding(design_id)
    field_map = field_module.design_field_map(mapping, binding)
    protocol, _ = build_protocol(design_id, domain, built=built, field_map=field_map, target_cell_m=target_cell_m, dt_s=dt_s)
    return protocol, mapping, field_map


def command_run(args: argparse.Namespace) -> int:
    if not args.allow_launch:
        print("[run] REFUSED: the design mini-sweep is a DRAFT (not preregistered); pass --allow-launch only for a labelled non-evidentiary shakedown", file=sys.stderr)
        return 2
    path = preflight_module.preflight_path(args.domain)
    if not path.is_file():
        print(f"[run] REFUSED: no whole-set preflight for {args.domain} ({path}); run `preflight --domain {args.domain}` first", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report["all_passed"]:
        print(f"[run] REFUSED: the whole-set preflight for {args.domain} did not pass every design", file=sys.stderr)
        return 2
    protocol, _, field_map = _prepare(args.design, args.domain, args.grid)
    results = results_dir(args.design, args.domain if args.grid == "50um" else f"{args.domain}-{args.grid}")
    results.mkdir(parents=True, exist_ok=True)
    protocol_path = results / "protocol.json"
    protocol_path.write_bytes(json.dumps(protocol, indent=1, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")
    runner = _runner()
    runner.run_steady_state(protocol, results, backend=args.backend, field_map=field_map, max_steps=args.max_steps, wall_budget_seconds=args.wall_budget_seconds,
                            require_same_code=not args.ignore_code_identity, protocol_path=protocol_path)
    return 0


def command_finalize(args: argparse.Namespace) -> int:
    protocol, _, field_map = _prepare(args.design, args.domain)
    results = results_dir(args.design, args.domain)
    _runner().finalize(protocol, results, backend=args.backend, field_map=field_map, stop_reason=args.stop_reason, protocol_path=results / "protocol.json",
                       allow_refinalize=args.allow_refinalize, recover_runner_stop=args.recover_runner_stop)
    return 0


def command_status(args: argparse.Namespace) -> int:
    protocol, _ = build_protocol(args.design, args.domain)
    _print_json(_runner().status(results_dir(args.design, args.domain), protocol))
    return 0


def command_targets(args: argparse.Namespace) -> int:
    results = results_dir(args.design, args.domain)
    maps_path, summary_path = results / "maps.npz", results / "summary.json"
    if not maps_path.is_file() or not summary_path.is_file():
        print(f"[targets] no finished run under {results}", file=sys.stderr)
        return 1
    built = design_module.build_design(args.design)
    mapping = design_module.pic_geometry(built, args.domain)
    binding = field_module.load_binding(args.design)
    topology = binding.get("topology_under_iron")
    if topology is not None:
        cusps = [c["z_c_m"] for c in topology["wall_cusps"]]
        cells = topology["cells"]
    else:
        entry = design_module.catalogue_entry(args.design)
        cusps = [c["z_c_m"] for c in entry["wall_cusps"]]
        cells = entry["cells"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    targets = extract_targets(load_maps(maps_path), mapping, cusps, cells, window_currents=summary.get("window_currents_a"))
    out = results / "closure-targets.json"
    out.write_bytes(json.dumps(targets, indent=1, sort_keys=True, allow_nan=True).encode("utf-8") + b"\n")
    print(f"[targets] wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, *, design: bool | None = None, domain: bool = False) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        if design is not None:
            p.add_argument("--design", required=design, default=None)
        if domain:
            p.add_argument("--domain", default="channel", choices=design_module.DOMAIN_OPTIONS)
        p.set_defaults(fn=fn)
        return p

    add("fields", command_fields, design=False)
    add("preflight", command_preflight, design=False, domain=True)
    protocol_parser = add("protocol", command_protocol, design=True, domain=True)
    protocol_parser.add_argument("--grid", default="50um", choices=tuple(GRID_VARIANTS), help="50um = template; 33um = the attempt-8 refinement variant (33.3 um / 1.4 ps)")
    add("draft-protocol", command_draft_protocol, domain=True)
    add("cost", command_cost)
    run_parser = add("run", command_run, design=True, domain=True)
    run_parser.add_argument("--grid", default="50um", choices=tuple(GRID_VARIANTS))
    run_parser.add_argument("--backend", default="warp-cuda")
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--wall-budget-seconds", type=float, default=None)
    run_parser.add_argument("--ignore-code-identity", action="store_true")
    run_parser.add_argument("--allow-launch", action="store_true", help="required; DRAFT guard (labelled non-evidentiary shakedowns only)")
    fin = add("finalize", command_finalize, design=True, domain=True)
    fin.add_argument("--backend", default="warp-cuda")
    fin.add_argument("--stop-reason", default="finalized_from_checkpoint")
    fin.add_argument("--allow-refinalize", action="store_true")
    fin.add_argument("--recover-runner-stop", action="store_true")
    add("status", command_status, design=True, domain=True)
    add("targets", command_targets, design=True, domain=True)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
