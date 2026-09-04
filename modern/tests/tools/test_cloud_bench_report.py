"""Unit tests for ``tools.cloud.bench_gpu_concurrency``: registry, overrides, aggregation and report formatting.

Synthetic worker timings only - no GPU, no Warp.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.cloud import bench_gpu_concurrency as bench

MODERN = Path(__file__).resolve().parents[2]


def _worker(index: int, ms: float, *, count: int, setup: float = 60.0, mib: float = 5000.0, ok: bool = True) -> dict:
    steps = 2000
    return {
        "config": "channel-50um", "index": index, "count": count, "ok": ok, "pid": 1000 + index,
        "ms_per_step_wall": ms, "measure_s": ms * steps / 1e3,
        "timing": {"steps": steps, "seconds": ms * steps / 1e3, "ms_per_step": ms, "accumulation": True},
        "timing_function": "experiments.pic2d_cft_steady_state_v4.run._time_steps",
        "simulation_construct_s": setup, "inputs_s": 3.0, "prewarm_s": 5.0,
        "nvidia_smi_used_mib_max": mib, "gpu_memory": {"mempool_used_mem_high": int(mib * 0.9 * 2**20)},
        "particles_initial": {"electrons": 260000, "ions": 260000}, "step_graph": True,
    }


# ------------------------------------------------------------------------------------------ registry / overrides
def test_registry_protocols_exist_and_default_set_is_the_measured_configurations() -> None:
    for config in bench.CONFIGS.values():
        assert (MODERN / config.protocol).is_file(), config.protocol
    assert bench.default_config_keys() == ["channel-50um", "channel-33um", "plume-v2.0-50um"]
    measured = {k for k, c in bench.CONFIGS.items() if not c.anchor.predicted}
    assert measured == {"channel-50um", "channel-33um", "plume-v2.0-50um"}
    assert bench.CONFIGS["channel-50um"].anchor.production_ms_per_step == 1.98
    assert bench.CONFIGS["channel-33um"].anchor == bench.Anchor(2.54, bench.CONFIGS["channel-33um"].anchor.seed_load_note,
                                                               4.36, bench.CONFIGS["channel-33um"].anchor.production_note)
    assert bench.CONFIGS["plume-v2.0-50um"].anchor.production_ms_per_step == pytest.approx(7.08)


def test_timing_function_is_the_v4_preflight_one_with_an_identical_fallback() -> None:
    fn, name = bench.timing_function()
    assert name == "experiments.pic2d_cft_steady_state_v4.run._time_steps"

    class FakeBackend:
        step_index = 0

    class FakeSim:
        def __init__(self) -> None:
            self.backend = FakeBackend()
            self.calls: list[tuple[int, int]] = []

        def run(self, steps, *, accumulate_from_step=None, progress=None):
            self.calls.append((steps, accumulate_from_step))

    for candidate in (fn, bench._time_steps_fallback):
        sim = FakeSim()
        out = candidate(sim, 40, warmup=10)
        assert sim.calls == [(10, 0), (40, 0)]           # warm-up then timed steps, accumulation from the start
        assert out["steps"] == 40 and out["accumulation"] is True and out["ms_per_step"] == pytest.approx(1e3 * out["seconds"] / 40)


def test_apply_overrides_is_a_deep_copy_with_dotted_paths() -> None:
    protocol = {"case": {"radial_cells": 60, "axial_cells": 480}, "numerics": {"dt_s": 1.5e-12}}
    result = bench.apply_overrides(protocol, {"case.radial_cells": 90, "numerics.dt_s": 1.4e-12, "new.key": 1})
    assert result["case"] == {"radial_cells": 90, "axial_cells": 480}
    assert result["numerics"]["dt_s"] == 1.4e-12 and result["new"] == {"key": 1}
    assert protocol["case"]["radial_cells"] == 60 and "new" not in protocol   # the frozen protocol is untouched


def test_load_bench_protocol_uses_the_v4_refinement_and_applies_the_production_load() -> None:
    protocol = bench.load_bench_protocol(bench.CONFIGS["channel-33um"], load="production")
    assert (protocol["case"]["radial_cells"], protocol["case"]["axial_cells"]) == (90, 720)
    assert protocol["numerics"]["dt_s"] == 1.4e-12
    assert protocol["case"]["macro_weight"] == pytest.approx(60000.0 / 2.25, rel=1e-4)
    assert protocol["operating_point"]["seed_plasma_density_per_m3"] == 1.75e17    # the v4 preflight's plateau load
    assert protocol["experiment_id"] == "pic2d-cft-steady-state-v4"
    ladder = bench.load_bench_protocol(bench.CONFIGS["channel-25um"])
    assert (ladder["case"]["radial_cells"], ladder["case"]["axial_cells"]) == (120, 960)
    assert ladder["numerics"]["dt_s"] == 1.0e-12 and ladder["case"]["macro_weight"] == 15000.0
    base = json.loads((MODERN / bench.CHANNEL_PROTOCOL).read_text(encoding="utf-8"))
    assert base["case"]["radial_cells"] == 60 and base["operating_point"]["seed_plasma_density_per_m3"] == 5e16
    seed = bench.load_bench_protocol(bench.CONFIGS["plume-v2.0-50um"], load="seed")
    assert seed["operating_point"]["seed_plasma_density_per_m3"] == base["operating_point"]["seed_plasma_density_per_m3"]
    extra = bench.load_bench_protocol(bench.CONFIGS["channel-50um"], {"case.macro_weight": 30000.0})
    assert extra["case"]["macro_weight"] == 30000.0


def test_parse_override_and_blas_pinning() -> None:
    assert bench.parse_override("case.radial_cells=90") == ("case.radial_cells", 90)
    assert bench.parse_override("case.id=\"x\"") == ("case.id", "x")
    assert bench.parse_override("case.id=plain") == ("case.id", "plain")
    with pytest.raises(Exception, match="KEY=VALUE"):
        bench.parse_override("novalue")
    assert bench.blas_threads_per_process(1, cpu_count=200) == 16
    assert bench.blas_threads_per_process(8, cpu_count=200) == 16
    assert bench.blas_threads_per_process(4, cpu_count=8) == 2
    assert bench.blas_threads_per_process(100, cpu_count=8) == 1
    env = bench.blas_environment(6)
    assert env == {"OMP_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6", "MKL_NUM_THREADS": "6", "NUMEXPR_NUM_THREADS": "6"}


# ------------------------------------------------------------------------------------------ aggregation
def test_summarise_round_and_speedups_with_synthetic_timings() -> None:
    config = bench.CONFIGS["channel-50um"]
    single = bench.summarise_round(config, 1, 0, 16, [_worker(0, 2.0, count=1)], [95.0, 97.0])
    double = bench.summarise_round(config, 2, 0, 8, [_worker(0, 3.0, count=2), _worker(1, 3.2, count=2)], [99.0])
    quad = bench.summarise_round(config, 4, 0, 4, [_worker(k, 6.0, count=4) for k in range(3)] + [_worker(3, 0.0, count=4, ok=False)], [])
    for r in (single, double, quad):
        r["load"] = "seed"
    assert single["aggregate_steps_per_s"] == pytest.approx(500.0)
    assert double["aggregate_steps_per_s"] == pytest.approx(1e3 / 3.0 + 1e3 / 3.2)
    assert double["overlap_fraction"] == pytest.approx(3.0 / 3.2)
    assert quad["processes_ok"] == 3 and quad["processes_failed"] == 1
    assert quad["gpu_memory_mib_per_process_nvidia_smi"] == [5000.0] * 3
    assert quad["gpu_memory_mib_per_process_warp_high"] == pytest.approx([4500.0] * 3)
    rounds = [single, double, quad]
    bench.attach_speedups(rounds)
    assert single["aggregate_speedup_vs_single_process"] == pytest.approx(1.0)
    assert double["aggregate_speedup_vs_single_process"] == pytest.approx((1e3 / 3.0 + 1e3 / 3.2) / 500.0)
    # seed-load anchor 2.26 ms/step on the 5090: one process here at 2.0 ms/step is 1.13x faster
    assert single["anchor_5090_ms_per_step"] == 2.26 and single["anchor_5090_predicted"] is False
    assert single["per_process_speedup_vs_5090"] == pytest.approx(2.26 / 2.0)
    assert single["gpu_throughput_speedup_vs_5090"] == pytest.approx(500.0 * 2.26 / 1e3)
    assert quad["per_process_speedup_vs_5090"] == pytest.approx(2.26 / 6.0)


def test_production_load_uses_the_production_anchor_and_predicted_flag() -> None:
    config = bench.CONFIGS["channel-25um"]
    r = bench.summarise_round(config, 1, 0, 16, [{**_worker(0, 8.0, count=1), "config": "channel-25um"}], [])
    r["load"] = "production"
    bench.attach_speedups([r])
    assert r["anchor_5090_ms_per_step"] == 17.3 and r["anchor_5090_predicted"] is True
    assert r["per_process_speedup_vs_5090"] == pytest.approx(17.3 / 8.0)
    r_seed = bench.summarise_round(config, 1, 0, 16, [{**_worker(0, 8.0, count=1), "config": "channel-25um"}], [])
    r_seed["load"] = "seed"
    bench.attach_speedups([r_seed])
    assert r_seed["anchor_5090_ms_per_step"] is None and r_seed["per_process_speedup_vs_5090"] is None
    # the v4 refinement anchors are measured: seed 2.54, plateau 4.36
    v4 = bench.summarise_round(bench.CONFIGS["channel-33um"], 1, 0, 16,
                               [{**_worker(0, 2.0, count=1), "config": "channel-33um"}], [])
    v4["load"] = "production"
    bench.attach_speedups([v4])
    assert v4["anchor_5090_ms_per_step"] == 4.36 and v4["anchor_5090_predicted"] is False
    assert v4["per_process_speedup_vs_5090"] == pytest.approx(4.36 / 2.0)


# ------------------------------------------------------------------------------------------ formatting
def _report(rounds: list[dict]) -> dict:
    return {"schema": bench.SCHEMA, "kind": "gpu-concurrency", "host": "lambda-h100", "utc": "2026-09-05T00:00:00Z",
            "git_head": "abcdef0123456789", "gpu_index": 0, "gpu": {"name": "NVIDIA H100 80GB HBM3", "driver_version": "580.65"},
            "warmup_steps": 400, "steps": 2000, "load": "seed", "anchor_gpu": bench.LOCAL_ANCHOR_GPU, "rounds": rounds}


def test_format_markdown_renders_one_row_per_round_with_markers() -> None:
    config = bench.CONFIGS["channel-50um"]
    single = bench.summarise_round(config, 1, 0, 16, [_worker(0, 2.0, count=1)], [95.0])
    double = bench.summarise_round(config, 2, 0, 8, [_worker(0, 3.0, count=2), _worker(1, 4.5, count=2)], [])
    pred = bench.summarise_round(bench.CONFIGS["channel-25um"], 1, 0, 16,
                                 [{**_worker(0, 8.0, count=1), "config": "channel-25um"}], [])
    for r in (single, double):
        r["load"] = "seed"
    pred["load"] = "production"
    rounds = [single, double, pred]
    bench.attach_speedups(rounds)
    text = bench.format_markdown(_report(rounds))
    lines = text.splitlines()
    assert lines[0].startswith("# PIC step-loop concurrency on one GPU - NVIDIA H100 80GB HBM3 (index 0)")
    assert "git `abcdef01`" in text and "400 warm-up + 2000 timed steps" in text
    table = [line for line in lines if line.startswith("| channel-")]
    assert len(table) == 3
    assert table[0].startswith("| channel-50um | 1 | 2.00 | 500.0 | 1.00 | 5000 | 4500 | 60.0 | 2.26 | 1.13 | 1.13 |")
    assert "| 3.00 / 4.50 |" in table[1]
    assert "| 17.30 (pred.) | 2.16 |" in table[2]          # predicted anchor 17.3 / 8.0 measured
    assert "Warning: at least one round had < 80 % overlap" in text      # 3.0 / 4.5 = 0.67
    # a round with no anchor renders dashes, a failed process is flagged
    failed = bench.summarise_round(config, 2, 0, 8, [_worker(0, 3.0, count=2), _worker(1, 0.0, count=2, ok=False)], [])
    failed["load"] = "seed"
    bench.attach_speedups([failed])
    row = next(line for line in bench.format_markdown(_report([failed])).splitlines() if line.startswith("| channel-"))
    assert "| 2 (1 failed) |" in row


def test_format_markdown_handles_an_empty_or_partial_report() -> None:
    text = bench.format_markdown({"rounds": []})
    assert text.startswith("# PIC step-loop concurrency on one GPU - unknown GPU (index None)")
    partial = {"config": "plume-v2.0-50um", "concurrency": 4, "processes_failed": 4, "processes_ok": 0,
               "ms_per_step_wall_per_process": [], "aggregate_steps_per_s": None, "setup_s_per_process": [float("nan")]}
    text = bench.format_markdown(_report([partial]))
    row = next(line for line in text.splitlines() if line.startswith("| plume-"))
    assert row.count(" - |") >= 8 and "(4 failed)" in row


def test_format_round_line_and_factorisation_table() -> None:
    config = bench.CONFIGS["channel-50um"]
    r = bench.summarise_round(config, 2, 0, 8, [_worker(0, 3.0, count=2), _worker(1, 3.0, count=2)], [])
    r["load"] = "seed"
    bench.attach_speedups([r])
    line = bench.format_round_line(r)
    assert line.startswith("[bench] channel-50um N=2: ms/step 3.00 / 3.00 aggregate 666.7 steps/s")
    report = {"kind": "host-factorisation", "host": "lambda", "cpu_count": 208, "rounds": [
        {"config": "plume-v2.1-50um", "concurrency": 8, "blas_threads_per_process": 16, "processes_failed": 0,
         "factorisation_s_per_process": [700.0, 720.5], "factorisation_s_max": 720.5, "block_size": 961, "blocks": 241,
         "inverse_blocks_gib": 1.66}]}
    text = bench.format_factorisation_markdown(report)
    assert "| plume-v2.1-50um | 8 | 16 | 700.0 / 720.5 | 720.5 | 961 | 241 | 1.66 |" in text
    assert "floor(CPUs / N)" in text


def test_report_command_roundtrip(tmp_path: Path, capsys) -> None:
    config = bench.CONFIGS["channel-50um"]
    r = bench.summarise_round(config, 1, 0, 16, [_worker(0, 2.0, count=1)], [])
    r["load"] = "seed"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_report([r])), encoding="utf-8")
    md = tmp_path / "r.md"
    assert bench.main(["report", str(path), "--markdown", str(md)]) == 0
    assert md.read_text(encoding="utf-8").count("| channel-50um |") == 1
    assert "channel-50um" in capsys.readouterr().out


def test_run_dry_run_lists_rounds_without_spawning(capsys) -> None:
    assert bench.main(["run", "--dry-run", "--gpu", "3", "--concurrency", "1", "2", "--configs", "channel-50um"]) == 0
    out = capsys.readouterr().out
    assert "would run channel-50um with N=1 on GPU 3" in out and "N=2 on GPU 3" in out
    with pytest.raises(SystemExit):
        bench.main(["run", "--dry-run", "--configs", "no-such-config"])
