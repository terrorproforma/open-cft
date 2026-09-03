"""v2.0 cathode placement on channel-connected field lines and the ignition gate (plume attempt 4)."""

from __future__ import annotations

import numpy as np
import pytest

from cft_revival.pic2d import fieldlines as fl
from cft_revival.pic2d.fields import MagneticFieldMap, uniform_field_map
from cft_revival.pic2d.mesh import build_mesh_masks
from cft_revival.pic2d.models import ChannelGeometry, Grid2D, PIC2DValidationError
from experiments.pic2d_cft_steady_state_v1 import run as runner


def _plume_grid() -> Grid2D:
    geometry = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 6.0e-3, 3.0e-3, plume_radius_m=6.0e-3, plume_length_m=4.0e-3, body_dielectric_radius_m=4.0e-3)
    return Grid2D(geometry, 24, 48)


def test_uniform_axial_field_lines_enter_the_channel_only_inside_the_aperture() -> None:
    grid = _plume_grid()
    masks = build_mesh_masks(grid)
    field = uniform_field_map(grid, 0.05)
    # inside the aperture radius, in the plume: one half-line enters the channel, the other leaves through z_max
    fwd, bwd = fl.trace_both(field, masks, 1.0e-3, 9.0e-3)
    assert {fwd.termination, bwd.termination} == {"channel", "far_field"}
    assert fwd.termination == "far_field" and fwd.points[-1, 1] >= grid.geometry.domain_z_max_m - 1e-9
    assert bwd.termination == "channel" and bwd.points[-1, 1] < grid.geometry.z_max_m
    assert fl.connects_to_channel(field, masks, 1.0e-3, 9.0e-3)[0]
    # outside the aperture radius the line ends on the front face (body) and the far field
    fwd, bwd = fl.trace_both(field, masks, 4.5e-3, 9.0e-3)
    assert {fwd.termination, bwd.termination} == {"body", "far_field"}
    assert not fl.connects_to_channel(field, masks, 4.5e-3, 9.0e-3)[0]
    # a start inside the channel reaches the anode against B
    fwd, bwd = fl.trace_both(field, masks, 1.0e-3, 4.0e-3)
    assert bwd.termination == "anode" and fwd.termination == "far_field"
    with pytest.raises(PIC2DValidationError):
        fl.trace_field_line(field, masks, 5.0e-3, 4.0e-3)  # start inside the thruster body


def test_radial_field_lines_hit_the_wall_and_annulus_connectivity_fractions() -> None:
    grid = _plume_grid()
    masks = build_mesh_masks(grid)
    axial = uniform_field_map(grid, 0.05)
    connected = fl.annulus_connectivity(axial, masks, 0.5e-3, 2.0e-3, 8.5e-3, 9.5e-3, n_r=4, n_z=3)
    assert connected["connected_fraction"] == 1.0 and connected["n"] == 12
    assert set(connected["terminations"]) == {"channel", "far_field"}
    detached = fl.annulus_connectivity(axial, masks, 4.5e-3, 5.5e-3, 9.0e-3, 10.0e-3, n_r=4, n_z=3)
    assert detached["connected_fraction"] == 0.0
    assert set(detached["terminations"]) == {"body", "far_field"}
    # a purely radial field in the channel ends on the dielectric wall / axis
    b_r = np.full(grid.node_shape, 0.05)
    b_r[0, :] = 0.0
    radial = MagneticFieldMap(grid, b_r, np.zeros(grid.node_shape), {"kind": "test"})
    line = fl.trace_field_line(radial, masks, 1.0e-3, 4.0e-3, direction=1)
    assert line.termination == "wall"
    null = MagneticFieldMap(grid, np.zeros(grid.node_shape), np.zeros(grid.node_shape), {"kind": "test"})
    assert fl.trace_field_line(null, masks, 1.0e-3, 4.0e-3).termination == "null"


def test_channel_connected_flux_tube_bands_follow_the_aperture_in_a_uniform_field() -> None:
    grid = _plume_grid()
    masks = build_mesh_masks(grid)
    tube = fl.channel_connected_flux_tube(uniform_field_map(grid, 0.05), masks, n_lines=8, z_probe_m=[9.0e-3, 11.0e-3])
    assert tube["terminations"] == {"far_field": 8}
    for band in tube["bands_by_probe_z_m"].values():
        assert band is not None and 0.0 < band[0] < band[1] < grid.geometry.exit_radius_m


def test_runner_cathode_connectivity_check_is_fail_closed() -> None:
    grid = _plume_grid()
    masks = build_mesh_masks(grid)
    field = uniform_field_map(grid, 0.05)
    good = {"operating_point": {"cathode": {"r_inner_m": 0.5e-3, "r_outer_m": 2.0e-3, "z_start_m": 8.5e-3, "z_end_m": 9.5e-3,
                                            "require_channel_connected_fraction": 1.0}}}
    summary = runner.cathode_connectivity_check(good, field, masks)
    assert summary is not None and summary["connected_fraction"] == 1.0 and summary["required_fraction"] == 1.0
    assert "channel_flux_tube" in summary and summary["samples"] == 24
    bad = {"operating_point": {"cathode": {"r_inner_m": 4.5e-3, "r_outer_m": 5.5e-3, "z_start_m": 9.0e-3, "z_end_m": 10.0e-3,
                                           "require_channel_connected_fraction": 1.0}}}
    with pytest.raises(PIC2DValidationError, match="not channel-connected"):
        runner.cathode_connectivity_check(bad, field, masks)
    # not declared -> not gated
    assert runner.cathode_connectivity_check({"operating_point": {"cathode": {"r_inner_m": 4.5e-3}}}, field, masks) is None
    assert runner.cathode_connectivity_check({"operating_point": {}}, field, masks) is None


def _series(t_end_s: float, s_of_t, n_of_t) -> dict[str, np.ndarray]:
    time_s = np.arange(0.0, t_end_s, 3.0e-9)
    return {"step": np.arange(time_s.size), "time_s": time_s, "current_ionization_rate_per_s": s_of_t(time_s), "electrons": n_of_t(time_s)}


RULE = {"ignition_gate": {"reference_window_s": [5e-8, 2e-7], "check_window_s": 1.5e-7,
                          "checks": [{"time_s": 7.5e-7, "min_s_ratio": 0.8, "min_electron_ratio": 1.1},
                                     {"time_s": 1.5e-6, "min_s_ratio": 1.2, "min_electron_ratio": 1.4}]}}


def test_ignition_gate_passes_a_growing_discharge_and_stops_a_decaying_one() -> None:
    # v1.3-like growth: S e-fold 2.8 us, N_e e-fold 1.9 us
    ignited = _series(1.6e-6, lambda t: 1.7e16 * np.exp(t / 2.8e-6), lambda t: 2.4e5 * np.exp(t / 1.9e-6))
    result = runner.evaluate_ignition(ignited, RULE)
    assert result is not None and not result["pending"] and not result["failed"] and result["reason"] is None
    assert [c["evaluated"] for c in result["checks"]] == [True, True] and all(c["passed"] for c in result["checks"])
    assert result["checks"][0]["s_ratio"] > 1.0 and result["checks"][1]["electron_ratio"] > 1.4
    # attempt-3-like decay: S falls, N_e decays -> fails at the first check with a reason
    decaying = _series(0.8e-6, lambda t: 3e15 * np.exp(-t / 0.4e-6), lambda t: 2.4e5 * np.exp(-t / 5e-6))
    result = runner.evaluate_ignition(decaying, RULE)
    assert result["failed"] and result["reason"].startswith("no ignition: at 0.75 us")
    assert result["checks"][0]["passed"] is False and result["checks"][1]["evaluated"] is False
    # flat N_e with S holding (v1.3 attempt 1 class) fails on the electron ratio
    flat = _series(0.8e-6, lambda t: np.full_like(t, 2.6e15), lambda t: np.full_like(t, 4.2e4))
    result = runner.evaluate_ignition(flat, RULE)
    assert result["failed"] and result["checks"][0]["electron_ratio"] == pytest.approx(1.0)


def test_ignition_gate_is_pending_before_the_reference_window_and_absent_without_a_block() -> None:
    early = _series(1.0e-7, lambda t: np.full_like(t, 1e15), lambda t: np.full_like(t, 1e5))
    result = runner.evaluate_ignition(early, RULE)
    assert result is not None and result["pending"] and not result["failed"]
    before_checks = _series(0.5e-6, lambda t: np.full_like(t, 1e15), lambda t: np.full_like(t, 1e5))
    result = runner.evaluate_ignition(before_checks, RULE)
    assert not result["pending"] and not result["failed"] and all(c["evaluated"] is False for c in result["checks"])
    assert runner.evaluate_ignition(early, {"plateau_threshold": 0.05}) is None


def test_plume_protocol_declares_the_connected_cathode_and_the_gate() -> None:
    from pathlib import Path
    protocol = runner.load_protocol(Path(runner.__file__).resolve().parents[1] / "pic2d_cft_plume_v1" / "protocol.json")
    cathode = protocol["operating_point"]["cathode"]
    assert cathode["require_channel_connected_fraction"] == 1.0
    assert cathode["r_outer_m"] <= protocol["geometry"]["exit_radius_m"] and cathode["z_start_m"] > protocol["geometry"]["z_max_m"]
    gate = protocol["stopping_rule"]["ignition_gate"]
    assert [c["time_s"] for c in gate["checks"]] == [7.5e-7, 1.5e-6]
