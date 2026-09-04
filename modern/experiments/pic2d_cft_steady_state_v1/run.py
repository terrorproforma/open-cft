"""Detached, checkpointed, resumable steady-state runner (models v1.2 / v1.3).

From ``modern/``::

    $env:PYTHONPATH="$PWD\\src;$PWD"
    python -m experiments.pic2d_cft_steady_state_v1.run run          # start, or resume if a checkpoint exists
    python -m experiments.pic2d_cft_steady_state_v1.run status       # last status line + projections
    python -m experiments.pic2d_cft_steady_state_v1.run finalize     # summary/maps/series from the checkpoint, no stepping

Detached launch (PowerShell, from ``modern/``)::

    Start-Process python -ArgumentList "-u","-m","experiments.pic2d_cft_steady_state_v1.run","run" `
        -WindowStyle Hidden -RedirectStandardOutput results\\run.log -RedirectStandardError results\\run.err

The run writes, under ``results/``:

* ``status.jsonl`` - one machine-readable line per 200-step diagnostic sync
  (t, steps, N_e, N_i, I_d, I_beam,i, peak-node / mean n_e, <T_e>, max omega_pe dt,
  cumulative wall time, ms/step, latest plateau evaluation; with the v1.3 neutral
  inventory also n_g, its fixed point, S and the effusion rate);
* ``series.jsonl`` - the full series record per sync (the source of ``series.npz``);
* ``checkpoint/checkpoint-latest.{json,npz}`` - rewritten atomically every
  ``checkpoint_every_steps`` (bitwise-resumable dynamical state incl. n_g);
* ``run_state.json`` - cumulative wall time, sessions, last checkpoint step; ``finished`` /
  ``stop_reason`` belong to the CURRENT session only (a resume or a finalize demotes the previous
  terminal block - stop reason, ``finalized_from_step``, ``finalization_recovery`` /
  ``finalization_error`` - into the ``history`` list and rewrites the file before its first step);
* ``run.pid`` - PID of the running process;
* on any stop: ``summary.json``, ``series.npz``, ``maps.npz``, ``checkpoint-final.*``.

Stop conditions: plateau (relative drift of I_d, N_e and - when present - n_g < 5 %
over the trailing 20 % of elapsed simulated time, only after >= 3 ion transit times),
the cumulative wall budget, the fail-closed stability gate, or an explicit
``--max-steps``.  Every stop exits 0 with the artifacts written.  The same module
drives ``pic2d_cft_steady_state_v2`` (model v1.3) through its own protocol.
Development/screening runs: not preregistered, no validated physics claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping

import numpy as np

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import MagneticFieldMap, build_p2_psi_field, p2_plume_field_map, sample_field_map
from cft_revival.pic2d.frames import FrameRecorder, FrameRecorderConfig, estimate_frame_bytes, frames_manifest, list_frames
from cft_revival.pic2d.mcc import MCCConfig, XenonCrossSections
from cft_revival.pic2d.models import (
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DStabilityError,
    PIC2DValidationError,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.coulomb import CoulombConfig
from cft_revival.pic2d.neutrals import NEUTRAL_LEDGER_KEYS, NeutralInventoryConfig
from cft_revival.pic2d.neutrals_spatial import NEUTRAL_SPATIAL_LEDGER_KEYS, MetastableConfig, SpatialNeutralConfig
from cft_revival.pic2d.see import SEEConfig
from cft_revival.pic2d.sensitivity import AnomalousCollisionConfig
from cft_revival.pic2d.simulation import (
    CathodeConfig,
    InjectionConfig,
    PeakDebyeGateConfig,
    PIC2DConfig,
    PlumeBoundaryGateConfig,
    SeedPlasmaConfig,
    SeriesRecord,
    Simulation,
    instantaneous_maps,
)
from experiments.pic2d_cft_snapshot_v1.run import _exit_areas, _file_sha256, git_head
from experiments.pic2d_cft_steady_state_v1.gpu_sampler import DEFAULT_INTERVAL_SECONDS, GpuUtilisationSampler

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
REPOSITORY_ROOT = MODERN.parent
PROTOCOL_PATH = HERE / "protocol.json"
RESULTS = HERE / "results"

ELEMENTARY_CHARGE_C = 1.602176634e-19
EPSILON_0 = 8.8541878128e-12
ELECTRON_MASS_KG = 9.1093837139e-31

SERIES_SCALARS = (
    "step", "time_s", "electrons", "ions", "phi_mean_v", "phi_min_v", "phi_max_v", "kinetic_electron_j",
    "kinetic_ion_j", "field_energy_j", "surface_charge_c", "peak_omega_pe_dt", "poisson_iterations",
)
LEDGER_SCALARS = (
    "total_energy_j", "interval_residual_j", "interval_sources_j", "interval_electrode_work_j",
    "interval_field_work_j", "anode_induced_charge_c", "exit_induced_charge_c",
)
# v2.0.6 (energy-ledger correction): the W-scaled inelastic sink per interval; absent from pre-v2.0.6 records -> NaN
LEDGER_SCALARS_V206 = ("interval_inelastic_loss_j",)
# v2.3.0 (xe_collision_set_v2): the ion-neutral energy sink and the fast-neutral exit energy per interval; absent -> NaN
LEDGER_SCALARS_V23 = ("interval_ion_neutral_loss_j", "interval_ke_fast_neutral_exit_j")
NEUTRAL_SCALARS_V23 = ("fast_neutral_exit_rate_per_s",)
NEUTRAL_LEDGER_KEYS_V23 = ("fast_neutral_exit",)
MOMENTUM_OPTIONAL_SCALARS_V23 = (
    "ion_collision_momentum_rate_n", "fast_neutral_exit_momentum_rate_n", "fast_neutral_wall_momentum_rate_n", "gas_momentum_rate_n",
    "fast_neutral_thrust_n", "fast_neutral_exit_power_w",
)
# v2.4.0 (coulomb_v1): the relativistic pair-energy tally of the Coulomb operator per interval (~0; absent -> NaN) and the
# Coulomb sample (records with the operator on only; see Simulation._coulomb_record)
LEDGER_SCALARS_V24 = ("interval_coulomb_ke_j",)
COULOMB_SCALARS = (
    "nu_ee_mean_per_s", "nu_ei_mean_per_s", "nu_ii_mean_per_s", "mean_s_ee", "mean_s_ei", "mean_s_ii", "fraction_large_s_ee",
    "fraction_large_s_ei", "mean_coulomb_log_ee", "mean_coulomb_log_ei", "interval_ee_pairs", "interval_ei_pairs", "interval_ii_pairs",
    "interval_cycles", "interval_pz_coulomb_kg_m_s", "interval_ke_coulomb_j", "nu_en_elastic_mean_per_s", "nu_ee_over_nu_en",
    "nu_e_spitzer_peak_per_s", "nu_e_spitzer_peak_over_nu_en",
)
MOMENTUM_OPTIONAL_SCALARS_V24 = ("coulomb_momentum_rate_n",)
# v2.2.0 SEE sample (records of an emitting wall only; see Simulation._see_record)
SEE_SCALARS = (
    "interval_impacts", "interval_emitted", "interval_ion_induced_emitted", "interval_effective_yield", "interval_mean_yield",
    "interval_clamped_impacts", "cumulative_effective_yield", "emission_current_a", "wall_impact_current_a", "backscattered_fraction",
    "mean_emitted_energy_ev", "emitted_power_w", "wall_potential_mean_v", "wall_potential_min_v", "wall_potential_max_v",
    "plasma_minus_wall_mean_v",
)
NEUTRAL_SCALARS = (
    "density_per_m3", "fixed_point_per_m3", "scale", "ionization_rate_per_s", "effusion_rate_per_s",
    "artificial_rate_per_s", "interval_ledger_residual_atoms",
)
# v1.4 (wall-ion recycling): absent from v1.3 records -> NaN in the arrays
NEUTRAL_SCALARS_V14 = ("recycled_rate_per_s", "wall_ion_absorption_rate_per_s", "gross_utilisation", "net_utilisation")
# v2.5.0 (neutrals_spatial_v1): the spatial model's record carries these instead of fixed_point / scale / artificial (NaN otherwise);
# its atom ledger keys (NEUTRAL_SPATIAL_LEDGER_KEYS) are stored as neutral_ledger_<key> next to the 0-D ones
NEUTRAL_SCALARS_V25 = (
    "atoms_ground", "atoms_metastable", "macro_neutrals", "macro_metastables", "density_max_per_m3", "axis_density_anode_per_m3",
    "axis_density_exit_per_m3", "fast_neutral_in_rate_per_s", "cex_converted_rate_per_s", "neutral_exit_thrust_n", "neutral_exit_power_w",
    "neutral_wall_force_n", "debt_ground_atoms", "debt_meta_atoms", "pending_atoms", "interval_meta_ledger_residual_atoms",
    "sink_consistency_atoms", "neutral_time_s", "time_acceleration", "ceiling_violation_fraction",
)
METASTABLE_SCALARS_V25 = (
    "channel_mean_density_per_m3", "fraction_of_ground", "production_rate_per_s", "stepwise_ionization_rate_per_s", "superelastic_rate_per_s",
    "wall_deexcitation_rate_per_s", "radiative_rate_per_s", "effusion_rate_per_s", "stepwise_fraction_of_ionization",
)
# v1.4 peak-node Debye sample (blocker 1); absent from v1.3 records -> arrays omitted
PEAK_NODE_SCALARS = (
    "n_e_peak_per_m3", "t_e_peak_ev", "debye_length_m", "cells_per_debye", "macro_particles_at_peak", "t_e_dense_ev",
    "r_m", "z_m",
)
# v2.0.3 window-mode peak-Debye gate (peak_node["window"]); NaN in single-step records -> arrays peak_node_window_<key>
# v2.0.6: + the accumulated particle-steps at the gated peak (NaN without the accumulated floor) and the v2.0.3
# occupancy-floor witness ``occupancy_floor_peak.cells_per_debye`` -> array peak_node_window_occupancy_floor_cells_per_debye
PEAK_NODE_WINDOW_SCALARS = (
    "cells_per_debye", "n_e_peak_per_m3", "t_e_peak_ev", "window_steps", "mean_macro_particles_at_peak", "resolved_nodes", "r_m", "z_m",
    "accumulated_macro_particle_steps_at_peak",
)
PEAK_NODE_WINDOW_WITNESS_SCALARS = ("occupancy_floor_cells_per_debye",)
# v2.0 momentum / thrust ledger and plume-boundary sample (absent from v1.x records -> arrays omitted)
MOMENTUM_SCALARS = (
    "momentum_z_kg_m_s", "interval_ledger_residual_kg_m_s", "beam_momentum_rate_ions_n", "beam_momentum_rate_electrons_n",
    "injected_momentum_rate_n", "absorbed_momentum_rate_n", "field_impulse_rate_n", "electric_impulse_rate_n",
    "magnetic_impulse_rate_n", "collision_momentum_rate_n", "born_momentum_rate_n", "dp_rate_n", "thrust_flux_n",
    "cold_gas_thrust_n", "thrust_total_n", "force_on_thruster_n", "thrust_balance_n", "closure_fraction",
    "electrostatic_force_thruster_n", "electrostatic_force_far_field_n",
)
MOMENTUM_OPTIONAL_SCALARS = ("cathode_target_rate_per_step", "cathode_rate_per_step", "cathode_emission_next_a")
PLUME_SCALARS = (
    "far_field_phi_max_abs_deviation_v", "far_field_net_charge_density_max_per_m3", "peak_electron_density_per_m3",
    "charge_fraction_of_peak", "far_field_induced_charge_c", "body_conductor_induced_charge_c", "exit_plane_axis_potential_v",
    "axis_phi_max_v", "axis_phi_max_z_m", "acceleration_z90_m", "acceleration_z10_m", "acceleration_width_m",
    "cathode_rate_per_step",
    # v2.0.1: unrestricted single-deposit statistic and the resolved-node count (NaN in attempt-6 and older records)
    "charge_fraction_of_peak_raw", "far_field_net_charge_density_max_raw_per_m3", "far_field_resolved_nodes",
    # v2.0.2: the gate quantity is the trailing-window average; its window length, the unrestricted window statistic,
    # the window-mean peak and the largest accumulated node weight (NaN in attempt-7/8 and older records)
    "far_field_window_steps", "charge_fraction_of_peak_window_raw", "peak_electron_density_window_per_m3",
    "far_field_accumulated_macro_particles_max",
)


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_variants(protocol_path: Path) -> dict[str, Any]:
    """Named variants from ``variants.json`` next to the protocol (empty if absent).

    They live outside ``protocol.json`` so a finished base run stays hash-bound to
    its (frozen) protocol file while convergence cases are added afterwards.
    """

    path = protocol_path.with_name("variants.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("variants", {})


def apply_case(protocol: dict[str, Any], case_name: str | None, variants: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """Return (protocol with the named variant merged into ``case``/``stopping_rule``, results dir name).

    A variant may override ``case`` keys (``id``, ``seed``, ``macro_weight``, cells) and
    ``wall_budget_seconds``.  ``None`` is the base case with results dir ``results``; a
    variant writes to ``results-<name>``.
    """

    if case_name is None:
        return protocol, "results"
    variants = dict(variants if variants is not None else protocol.get("variants") or {})
    if case_name not in variants:
        raise PIC2DValidationError(f"unknown case {case_name!r}; known: {sorted(variants)}")
    variant = variants[case_name]
    merged = json.loads(json.dumps(protocol))
    merged["case"] = {**merged["case"], **{k: v for k, v in variant.items() if k in ("id", "seed", "macro_weight", "radial_cells", "axial_cells")}}
    if "wall_budget_seconds" in variant:
        merged["stopping_rule"]["wall_budget_seconds"] = variant["wall_budget_seconds"]
    merged["case"]["variant"] = case_name
    merged["case"]["variant_note"] = variant.get("note")
    return merged, f"results-{case_name}"


def protocol_budget(protocol: dict[str, Any]) -> dict[str, Any]:
    """The ``budget_v1_x`` block (one per protocol)."""

    keys = [key for key in protocol if key.startswith("budget")]
    if len(keys) != 1:
        raise PIC2DValidationError("protocol must carry exactly one budget block")
    return protocol[keys[0]]


def poisson_config(numerics: Mapping[str, Any], *, backend: str) -> PoissonConfig2D:
    """Field-solve selection (part of ``config_sha256``).

    Default (every protocol up to v2.0.4, whose ``numerics.poisson`` is a descriptive string): the exact block-Thomas
    solve - on the device for ``warp-cuda``, on the host otherwise - at the 1e-10 relative residual contract; identity
    unchanged.  poisson_gmg_v1: a ``numerics.poisson`` OBJECT ``{"method": "device-mg", "cycles": 14, "pre_sweeps": 2,
    "post_sweeps": 2, "omega": 0.8, "coarsest_max_unknowns": 1024}`` selects the fixed-cycle geometric multigrid
    (``cft_revival.pic2d.warp_poisson_mg``; the CPU backends run the same cycles in numpy).  A protocol naming a
    different solver is a different configuration identity, so a checkpoint never crosses solvers silently.
    """

    block = numerics.get("poisson")
    if not isinstance(block, Mapping) or "method" not in block:
        return PoissonConfig2D(method="device-direct" if backend == "warp-cuda" else "direct", relative_tolerance=1.0e-10)
    method = str(block["method"])
    if method == "device-direct" and backend != "warp-cuda":
        method = "direct"
    return PoissonConfig2D(
        method=method,   # type: ignore[arg-type]
        relative_tolerance=float(block.get("relative_tolerance", 1.0e-10)),
        mg_cycles=int(block.get("cycles", 14)),
        mg_pre_sweeps=int(block.get("pre_sweeps", 2)),
        mg_post_sweeps=int(block.get("post_sweeps", 2)),
        mg_omega=float(block.get("omega", 0.8)),
        mg_coarsest_max_unknowns=int(block.get("coarsest_max_unknowns", 1024)),
    )


def build_config(protocol: dict[str, Any], *, backend: str = "warp-cuda") -> PIC2DConfig:
    geometry = protocol["geometry"]
    case = protocol["case"]
    operating = protocol["operating_point"]
    numerics = protocol["numerics"]
    # v2.0: optional plume box (L-shaped plasma domain) declared by three extra geometry keys
    plume_keys = {key: float(geometry[key]) for key in ("plume_radius_m", "plume_length_m", "body_dielectric_radius_m") if geometry.get(key) is not None}
    grid = Grid2D(
        ChannelGeometry(
            geometry["bore_radius_m"], geometry["z_min_m"], geometry["z_max_m"],
            geometry["cone_start_z_m"], geometry["exit_radius_m"], **plume_keys,
        ),
        int(case["radial_cells"]), int(case["axial_cells"]),
    )
    sync = int(numerics["device_sync_steps"])
    checkpoint_every = int(numerics["checkpoint_every_steps"])
    window = int(numerics["averaging_window_steps"])
    if checkpoint_every % sync != 0 or window % checkpoint_every != 0:
        raise PIC2DValidationError("checkpoint_every_steps must be a multiple of device_sync_steps and divide averaging_window_steps")
    mcc = None
    if operating["neutral_density_per_m3"] > 0:
        # v2.3.0: an optional ``operating_point.collision_set`` block selects the hash-bound xenon collision set
        # (``xe_collision_set_v2``: four excitation levels + Xe+ / Xe CEX and MEX); absent = the legacy lumped set with
        # collisionless ions, whose config_sha256 is unchanged
        collision_set = None
        if operating.get("collision_set") is not None:
            from cft_revival.pic2d.cross_sections_xe import CollisionSetConfig

            collision_set = CollisionSetConfig.from_protocol(operating["collision_set"])
        mcc = MCCConfig(operating["neutral_density_per_m3"], operating["neutral_temperature_k"], collision_set=collision_set)
    inventory = None
    if operating.get("neutral_inventory") is not None:
        block = operating["neutral_inventory"]
        # v1.4: relaxation_time_s may be null (physical effusion time scale only) and the
        # inventory may recycle wall/anode ions as thermal neutrals at the wall temperature
        relaxation = block["relaxation_time_s"]
        inventory = NeutralInventoryConfig(
            float(block["feed_atoms_per_s"]), None if relaxation is None else float(relaxation),
            wall_recycling=bool(block.get("wall_recycling", False)),
            recombination_coefficient=float(block.get("recombination_coefficient", 1.0)),
            wall_temperature_k=None if block.get("wall_temperature_k") is None else float(block["wall_temperature_k"]),
            # v2.0: declared start density below the null-collision ceiling (headroom for the recycling transient)
            initial_density_per_m3=None if block.get("initial_density_per_m3") is None else float(block["initial_density_per_m3"]),
        )
        if int(numerics["series_interval_steps"]) != sync:
            raise PIC2DValidationError("the neutral inventory is updated at the series interval, which must equal device_sync_steps")
    # v2.5.0: ``operating_point.neutrals`` = {"model": "neutrals_spatial_v1", ...SpatialNeutralConfig fields, "metastables": {...} | null}
    # replaces the 0-D inventory (the two blocks are mutually exclusive); absent = inventory-0d, identity unchanged
    spatial = None
    if operating.get("neutrals") is not None:
        spatial = spatial_neutral_config_from_protocol(operating["neutrals"])
    peak_gate = None
    if numerics.get("peak_debye_gate") is not None:
        peak_gate = PeakDebyeGateConfig(**{k: v for k, v in numerics["peak_debye_gate"].items() if not k.endswith("_note")})
    anomalous = None
    if numerics.get("anomalous_collisions") is not None:
        anomalous = AnomalousCollisionConfig(**{k: v for k, v in numerics["anomalous_collisions"].items() if not k.endswith("_note")})
    # v2.2.0: secondary electron emission from the dielectric wall (numerics.see = SEEConfig fields; absent = the v2.0.x
    # absorbing wall and an unchanged config identity)
    see = None
    if numerics.get("see") is not None:
        see = SEEConfig(**{k: v for k, v in numerics["see"].items() if not k.endswith("_note")})
    # v2.4.0: Coulomb collisions (numerics.coulomb = CoulombConfig fields; absent = collisionless charged species and an
    # unchanged config identity)
    coulomb = None
    if numerics.get("coulomb") is not None:
        coulomb = CoulombConfig(**{k: v for k, v in numerics["coulomb"].items() if not k.endswith("_note")})
    # v2.0: a cathode emission region in the plume replaces the exit-plane injection (kept as the legacy option)
    cathode = None
    injection = None
    if operating.get("cathode") is not None:
        cathode = CathodeConfig(**{k: v for k, v in operating["cathode"].items() if not k.endswith("_note") and not k.endswith("_justification") and k != "require_channel_connected_fraction"})
    if operating.get("electron_injection_current_a") is not None:
        injection = InjectionConfig(operating["electron_injection_current_a"], operating["electron_injection_temperature_ev"])
    plume_gate = None
    if numerics.get("plume_boundary_gate") is not None:
        plume_gate = PlumeBoundaryGateConfig(**{k: v for k, v in numerics["plume_boundary_gate"].items() if not k.endswith("_note")})
    # v2.0.5: optional performance block - electron-moment sampling interval K (absent = 1 = every accumulated step,
    # the v2.0.x identity); K must divide the sync interval so every frame / window / checkpoint boundary (all multiples
    # of device_sync_steps, and the accumulators are reset at window boundaries) holds a whole number of samples
    performance = numerics.get("performance") or {}
    moment_sample_interval = int(performance.get("moment_sample_interval", 1))
    if moment_sample_interval < 1 or sync % moment_sample_interval != 0:
        raise PIC2DValidationError("performance.moment_sample_interval must be a positive divisor of device_sync_steps")
    return PIC2DConfig(
        grid=grid,
        potentials=BoundaryPotentials(operating["anode_potential_v"], operating["exit_plane_potential_v"]),
        dt_s=float(numerics["dt_s"]),
        macro_weight=float(case["macro_weight"]),
        seed=int(case.get("seed", 20260903)),
        injection=injection,
        cathode=cathode,
        plume_boundary_gate=plume_gate,
        seed_plasma=SeedPlasmaConfig(
            operating["seed_plasma_density_per_m3"], operating["seed_electron_temperature_ev"], operating["seed_ion_temperature_ev"],
            region=operating.get("seed_region", "all"),   # v2.0: "channel" leaves the plume empty at t = 0
        ),
        mcc=mcc,
        poisson=poisson_config(numerics, backend=backend),
        limits=StabilityLimits(**numerics["stability_limits"]),
        reference_density_per_m3=numerics["stability_reference"]["density_per_m3"],
        reference_electron_temperature_ev=numerics["stability_reference"]["electron_temperature_ev"],
        max_electron_energy_ev=numerics["stability_reference"]["max_electron_energy_ev"],
        series_interval_steps=int(numerics["series_interval_steps"]),
        runtime_stability_check_steps=sync,
        ion_subcycle=int(numerics["ion_subcycle"]),
        device_sync_steps=sync,
        neutral_inventory=inventory,
        peak_debye_gate=peak_gate,
        anomalous=anomalous,
        see=see,
        coulomb=coulomb,
        moment_sample_interval=moment_sample_interval,
        neutrals_spatial=spatial,
    )


def spatial_neutral_config_from_protocol(block: Mapping[str, Any]) -> SpatialNeutralConfig:
    """v2.5.0: the ``operating_point.neutrals`` block (``*_note`` / ``*_justification`` keys are documentation)."""

    def clean(mapping: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in mapping.items() if not k.endswith("_note") and not k.endswith("_justification") and k != "model"}

    model = block.get("model")
    if model != "neutrals_spatial_v1":
        raise PIC2DValidationError(f"unknown neutral model {model!r} (known: 'neutrals_spatial_v1'; the 0-D inventory is operating_point.neutral_inventory)")
    fields = clean(block)
    meta_block = fields.pop("metastables", None)
    metastables = None
    if meta_block is not None:
        if meta_block.get("model", "metastables_v1") != "metastables_v1":
            raise PIC2DValidationError(f"unknown metastable model {meta_block.get('model')!r}")
        meta_fields = clean(meta_block)
        meta_fields["branching"] = tuple(float(b) for b in meta_fields["branching"])
        metastables = MetastableConfig(**meta_fields)
    fields["substep_steps"] = int(fields["substep_steps"])
    return SpatialNeutralConfig(**fields, metastables=metastables)


def step_graph_flag(protocol: dict[str, Any]) -> bool:
    """v1.4: CUDA-graph replay of the step (``numerics.step_graph``, default on; no effect on the CPU backends)."""

    return bool(protocol["numerics"].get("step_graph", True))


def frame_recorder_config(protocol: dict[str, Any]) -> FrameRecorderConfig | None:
    """v2.0: ``numerics.frame_recorder`` = {cadence_steps, precision} (absent/null = OFF, the v1.x behaviour)."""

    block = protocol["numerics"].get("frame_recorder")
    if block is None:
        return None
    return FrameRecorderConfig(int(block["cadence_steps"]), str(block.get("precision", "float32")))


# -- plateau criterion ------------------------------------------------------

def trailing_time_drift(time_s: np.ndarray, values: np.ndarray, fraction: float) -> float | None:
    """Relative drift of a linear fit over the trailing ``fraction`` of the elapsed time.

    drift = slope * window / |mean|; ``None`` if fewer than 8 samples fall in the window
    or the mean is not usable.
    """

    if time_s.size < 8:
        return None
    t_end = float(time_s[-1])
    start = t_end - fraction * t_end
    mask = time_s >= start
    if int(mask.sum()) < 8:
        return None
    x = time_s[mask].astype(np.float64)
    y = values[mask].astype(np.float64)
    mean = float(np.mean(y))
    if not np.isfinite(mean) or abs(mean) < 1e-300:
        return None
    slope = float(np.polyfit(x - x[0], y, 1)[0])
    return slope * float(x[-1] - x[0]) / abs(mean)


ARMING_SETTLE_QUANTITIES: dict[str, str] = {"discharge_current": "current_discharge_a", "electron_count": "electrons"}


def drift_members_arming(arrays: dict[str, np.ndarray], rule: Mapping[str, Any], transit_time_s: float) -> dict[str, Any] | None:
    """Model v2.1.1: arming of the triad's DRIFT members relative to the run's own discharge (a "settled once" latch).

    ``stopping_rule.grid_heating_triad.drift_members_arming`` (absent -> ``None``: the v1.4 rule
    ``enforced_after_transit_times`` stands) declares ``min_transit_times`` (2.0), ``settle_quantity``
    (``discharge_current`` -> the trailing-20 % drift of I_d, or ``electron_count``), ``settle_drift_max`` (the
    plateau threshold, 0.05) and ``settle_check_cadence_steps`` (the checkpoint cadence).  The drift members are
    ENFORCED (hard 25 % stop) only once (i) at least ``min_transit_times`` have elapsed AND (ii) the latch has
    closed: at some record on the check cadence at or after ``min_transit_times`` the settle quantity's
    trailing-window drift read below ``settle_drift_max`` - the discharge has settled once, so a later drift of
    S, T_e,dense or omega_pe dt beyond the hard bound is a runaway, not the discharge still moving to its state.

    Why (alpha-series launch 1, 2026-09-05 05:00 AEST): the drift members were calibrated on the alpha = 0 plateaus
    (I_d drift +0.116 at 1.0 transit, S +0.10, T_e,dense +0.02 on ss-v4) and armed at 1.0 transit; a closure or
    operating-point change makes the discharge re-equilibrate to a DIFFERENT state, whose trailing-20 % drifts at
    1.0 transit can legitimately exceed 0.25 while nothing is wrong numerically.  The physics protections - the
    one-sided windowed residual-POWER gate (>= 5 % of the electrode work, from the first complete window) and the
    window-mode peak-Debye hard gate (pi cells per lambda_D on the accumulated-floor peak) - are what catch genuine
    finite-grid heating; they are independent of this arming (ss-v4 read +1.15 % / 0.48 cells per lambda_D at its
    1.00-transit stop under the old rule - nothing to protect against).  A discharge that never settles (an
    extinction: the alpha = 1/16 launch 1 decayed from its seed with N_e e-fold 0.88 us and I_d -> 0.06 mA) must be
    stopped by an ``ignition_gate`` (S / N_e ratios against the post-seed reference window), not by the drift
    members - the arming block is declared together with one.

    The latch is a pure function of the series (evaluated on the records whose step is a multiple of the declared
    cadence, exactly the checkpoints the runner evaluates at), so a resume or an offline re-read reproduces it.
    On the accepted alpha = 0 plateau (ss-v4) the I_d latch closes at 2.66 transits (checkpoint 4 560 000, drift +0.049);
    047 / 009 / 056-L2 read |I_d drift| < 0.05 from 2.0 transits on.
    """

    block = rule.get("grid_heating_triad") or {}
    arming = block.get("drift_members_arming")
    if arming is None or arrays.get("step") is None or arrays["step"].size < 2:
        return None
    min_transits = float(arming.get("min_transit_times", 2.0))
    settle_max = float(arming.get("settle_drift_max", rule.get("plateau_threshold", 0.05)))
    quantity = str(arming.get("settle_quantity", "discharge_current"))
    if quantity not in ARMING_SETTLE_QUANTITIES:
        raise PIC2DValidationError(f"drift_members_arming.settle_quantity {quantity!r} not in {sorted(ARMING_SETTLE_QUANTITIES)}")
    cadence = int(arming["settle_check_cadence_steps"])
    if cadence <= 0:
        raise PIC2DValidationError("drift_members_arming.settle_check_cadence_steps must be positive")
    fraction = float(rule["plateau_window_fraction"])
    time_s = arrays["time_s"]
    steps = arrays["step"]
    values = arrays[ARMING_SETTLE_QUANTITIES[quantity]].astype(np.float64)
    transits_now = float(time_s[-1]) / transit_time_s
    result: dict[str, Any] = {
        "latched": False, "armed": False, "transit_times_elapsed": transits_now, "min_transit_times": min_transits,
        "settle_quantity": quantity, "settle_drift_max": settle_max, "check_cadence_steps": cadence,
        "current_settle_drift": trailing_time_drift(time_s, values, fraction),
        "latched_at_step": None, "latched_at_transit_times": None, "drift_at_latch": None,
    }
    # the first checkpoint-cadence record at or after min_transit_times whose trailing-window drift reads inside the bound
    eligible = np.flatnonzero((time_s / transit_time_s >= min_transits) & (np.mod(steps, cadence) == 0))
    for index in eligible:
        n = int(index) + 1
        drift = trailing_time_drift(time_s[:n], values[:n], fraction)
        if drift is not None and abs(drift) < settle_max:
            result.update({"latched": True, "latched_at_step": int(steps[index]), "latched_at_transit_times": float(time_s[index]) / transit_time_s,
                           "drift_at_latch": float(drift)})
            break
    result["armed"] = bool(result["latched"] and transits_now >= min_transits)
    return result


def evaluate_plateau(
    time_s: np.ndarray, discharge_a: np.ndarray, electrons: np.ndarray, rule: dict[str, Any], transit_time_s: float,
    neutral_density: np.ndarray | None = None,
) -> dict[str, Any]:
    """The stopping rule: every tracked drift below the threshold AND >= min_transit_times elapsed.

    Tracked: discharge current and electron count; with the v1.3 inventory also n_g.
    """

    fraction = float(rule["plateau_window_fraction"])
    threshold = float(rule["plateau_threshold"])
    min_transits = float(rule["min_transit_times"])
    elapsed = float(time_s[-1]) if time_s.size else 0.0
    transits = elapsed / transit_time_s
    drifts = {
        "discharge_current_drift": trailing_time_drift(time_s, discharge_a, fraction),
        "electron_count_drift": trailing_time_drift(time_s, electrons, fraction),
    }
    if neutral_density is not None:
        drifts["neutral_density_drift"] = trailing_time_drift(time_s, neutral_density, fraction)
    drifts_ok = all(value is not None and abs(value) < threshold for value in drifts.values())
    return {
        "reached": bool(drifts_ok and transits >= min_transits),
        "drifts_within_threshold": bool(drifts_ok),
        "transit_times_elapsed": transits,
        "min_transit_times": min_transits,
        **drifts,
        "threshold": threshold,
        "window_fraction": fraction,
        "tracked": sorted(drifts),
    }


def evaluate_ignition(arrays: dict[str, np.ndarray], rule: dict[str, Any]) -> dict[str, Any] | None:
    """v2.0 ignition gate (fail-closed, ``stop_reason = no_ignition``).

    Reference: the means of the ionisation rate S and of the macro-electron count N_e over
    ``reference_window_s`` = [t0, t1] (after the seed dump, before growth).  At each declared
    check time t_c (once the series reaches it) the trailing ``check_window_s`` means must
    satisfy S/S_ref >= min_s_ratio and N_e/N_ref >= min_electron_ratio.  Calibrated on the
    v1.3 channel-only runs and the plume attempt 3: the ignited v1.3 attempt 2 (and seed-b)
    had S ratios 1.07 / 1.41 and N_e ratios 1.29 / 1.76 at 0.75 / 1.5 us; the failed v1.3
    attempt 1 had 0.59 / - and 1.03 / -; the plume attempt 3 (cathode not connected) had 0.23
    at 0.75 us with N_e 0.83.  A "x3 in 0.75 us" rule would have rejected the ignited run (its
    early S e-fold was ~2.8 us; the 1.1 us e-fold was N_e later) - hence the two-stage rule.
    Returns None when the protocol declares no ``ignition_gate`` block.
    """

    block = rule.get("ignition_gate")
    if block is None or arrays.get("step") is None or arrays["step"].size < 2:
        return None
    time_s = arrays["time_s"]
    s_rate = arrays["current_ionization_rate_per_s"]
    electrons = arrays["electrons"].astype(np.float64)
    t0, t1 = (float(v) for v in block["reference_window_s"])
    check_window = float(block.get("check_window_s", 0.15e-6))
    ref_mask = (time_s >= t0) & (time_s < t1)
    result: dict[str, Any] = {"reference_window_s": [t0, t1], "check_window_s": check_window, "checks": [], "failed": False, "reason": None}
    if not ref_mask.any() or float(time_s[-1]) < t1:
        result["pending"] = True
        return result
    s_ref = float(s_rate[ref_mask].mean())
    n_ref = float(electrons[ref_mask].mean())
    result.update({"s_reference_per_s": s_ref, "electrons_reference": n_ref, "pending": False})
    for check in block["checks"]:
        tc = float(check["time_s"])
        entry: dict[str, Any] = {"time_s": tc, "min_s_ratio": float(check["min_s_ratio"]), "min_electron_ratio": float(check["min_electron_ratio"])}
        if float(time_s[-1]) < tc:
            entry["evaluated"] = False
            result["checks"].append(entry)
            continue
        mask = (time_s >= tc - check_window) & (time_s <= tc)
        s_ratio = float(s_rate[mask].mean()) / s_ref if s_ref > 0.0 else 0.0
        n_ratio = float(electrons[mask].mean()) / n_ref if n_ref > 0.0 else 0.0
        entry.update({"evaluated": True, "s_ratio": s_ratio, "electron_ratio": n_ratio,
                      "passed": bool(s_ratio >= entry["min_s_ratio"] and n_ratio >= entry["min_electron_ratio"])})
        result["checks"].append(entry)
        if not entry["passed"] and not result["failed"]:
            result["failed"] = True
            result["reason"] = (f"no ignition: at {tc*1e6:.2f} us S/S_ref = {s_ratio:.2f} (min {entry['min_s_ratio']}), "
                                f"N_e/N_ref = {n_ratio:.2f} (min {entry['min_electron_ratio']})")
    return result


def cathode_connectivity_check(protocol: dict[str, Any], field_map: MagneticFieldMap, masks: Any) -> dict[str, Any] | None:
    """v2.0: fail-closed check that the cathode region sits on field lines entering the channel.

    ``operating_point.cathode.require_channel_connected_fraction`` (absent = not gated): the
    fraction of a uniform sample of the emission region whose field line (either direction)
    crosses the exit plane into the bore, from the event-aware tracer on the run's own node
    field.  Attempt 3's annulus (r 4.5-6 mm, z 26-28 mm) had fraction 0: its lines ran from the
    front face to the far field and 95 % of the emitted current left through the far field.
    """

    cathode = protocol["operating_point"].get("cathode")
    if cathode is None or cathode.get("require_channel_connected_fraction") is None:
        return None
    from cft_revival.pic2d.fieldlines import annulus_connectivity, channel_connected_flux_tube
    required = float(cathode["require_channel_connected_fraction"])
    result = annulus_connectivity(field_map, masks, float(cathode["r_inner_m"]), float(cathode["r_outer_m"]), float(cathode["z_start_m"]),
                                  float(cathode["z_end_m"]), n_r=6, n_z=4)
    tube = channel_connected_flux_tube(field_map, masks, n_lines=16)
    summary = {
        "required_fraction": required, "connected_fraction": result["connected_fraction"], "terminations": result["terminations"], "samples": result["n"],
        "channel_flux_tube": {"terminations": tube["terminations"], "bands_by_probe_z_m": tube["bands_by_probe_z_m"]},
        "method": "event-aware field-line tracing (RK2, 1/4 cell) on the bilinear node field; a sample is connected when either half-line "
                  "crosses the exit plane inside the aperture",
    }
    if result["connected_fraction"] < required:
        raise PIC2DValidationError(f"cathode region is not channel-connected: fraction {result['connected_fraction']:.2f} < {required} "
                                   f"(terminations {result['terminations']})")
    return summary


def evaluate_triad(arrays: dict[str, np.ndarray], rule: dict[str, Any], transit_time_s: float) -> dict[str, Any] | None:
    """v1.4 grid-heating triad (literature review, blocker 1, change (d)3), recorded and gated.

    (i) energy-ledger residual over the electrode work (cumulative; the momentum-conserving
    scheme heats the grid when the cells are coarser than lambda_D: Birdsall and Maron 1980,
    Ueda et al. 1994, Adams et al. 2025); (ii) T_e in the densest cells (n >= dense_fraction
    n_peak) and the ionisation rate S: their trailing-window drifts; (iii) the peak omega_pe dt
    drift.  Each drift uses the plateau window; ``soft`` thresholds must hold for a plateau to
    be declared; the ``hard`` thresholds (and the residual bound) stop the run fail-closed once
    ``enforced_after_transit_times`` have elapsed (before that the ratio and drifts are
    ill-conditioned: small electrode work, ignition transient).  Returns ``None`` when the
    protocol declares no triad block (v1.2/v1.3 protocols).

    v2.0.3 (``residual_window_steps`` in the block): the HARD residual member becomes the
    trailing-window residual power - the ledger residual summed over the records of the trailing
    ``residual_window_steps`` (the 400 000-step averaging window) divided by the electrode work
    over the same records - and it is one-sided: a POSITIVE residual is energy the scheme
    created (finite-grid heating); it stops the run once the window is complete and the ratio
    reaches ``windowed_energy_residual_over_electrode_work_max`` (0.05).  Plume attempt 8
    (2026-09-04): the per-window ratio went -0.5 % (2.0-2.4 us) -> +2.4 -> +5.8 (2.8-3.2 us)
    -> ... -> +54.8 % while the CUMULATIVE ratio, which lags by the whole history, was still
    +8.6 % (below its 10 % bound) when the S-drift member stopped the run at 4.98 us, ~1.8 us
    after the window ratio had crossed 5 %.  The accepted channel-only plateau runs (v2 base,
    seed-b, W x 0.7) have NEGATIVE window ratios throughout (-12.7 % in the seed window rising to
    -0.2 % / -1.4 % / -4.2 % at the plateau; max +0.37 %), so the negative side is recorded, not
    gated (a two-sided 5 % bound would have stopped all three accepted runs before 4 us).  The
    cumulative ratio stays recorded as the witness (``energy_residual_over_electrode_work``) and
    enters the soft (plateau-precondition) check against its bound as before, but no longer stops
    the run.  Without ``residual_window_steps`` the v1.4 cumulative hard gate is unchanged.

    v2.0.6 (2026-09-05, energy-ledger correction): the residual this member reads was biased NEGATIVE by the
    inelastic power up to v2.0.5 (``inelastic_loss_j`` lacked the macro weight; the recorded residual was
    ``H - L_inel`` with ``H`` the true numerical energy creation).  The calibration above was made on the biased
    statistic.  On the corrected statistic (``cft_revival.pic2d.ledger_recompute``, sidecars ``ledger-corrected.json``)
    the accepted 33 um plateaus read +0.6 % (056 L1), +0.9 % (047) and +2.5 % (ss-v4, still rising) in their
    end-state windows with maxima below 2.5 %, the 25 um v5 launch 1 +0.3 %; the accepted 50 um channel plateaus
    read +7.2 % (W x 0.7), +11.1 % (seed-b) and +13.0 % (base) - they were heating and this gate at 5 % would have
    stopped them at 4.5 / 2.8 / 2.7 us; plume attempts 6-8 read +12.9 % in their FIRST complete window (0.66 us)
    and attempt 8 never below +4.1 %; the external-validation launch 1 crossed 5 % at 0.34 us (recorded: 0.73 us).
    The thresholds (hard 5 % from the first complete window, one-sided) are kept: 2x margin over the accepted 33 um
    maxima, and every heating run is caught at its first complete window or earlier than under the biased statistic.
    From v2.0.6 the series carries the corrected residual directly, so this function needs no change.

    v2.1.1 (``drift_members_arming`` in the block, see :func:`drift_members_arming`): the DRIFT members are enforced
    only once ``min_transit_times`` have elapsed AND the settle latch has closed (the run's own I_d drift has read
    inside the plateau bound once); ``enforced_after_transit_times`` is then superseded (recorded, not used).  The
    windowed residual-power member is unchanged: enforced from the first complete window whatever the arming.
    """

    block = rule.get("grid_heating_triad")
    if block is None or arrays.get("step") is None or arrays["step"].size < 2:
        return None
    fraction = float(rule["plateau_window_fraction"])
    time_s = arrays["time_s"]
    transits = float(time_s[-1]) / transit_time_s
    residual = float(arrays["interval_residual_j"][1:].sum())
    electrode = float(arrays["interval_electrode_work_j"][1:].sum())
    ratio = residual / electrode if abs(electrode) > 0.0 else None
    drifts = {
        "ionisation_rate_drift": trailing_time_drift(time_s, arrays["current_ionization_rate_per_s"], fraction),
        "omega_pe_dt_drift": trailing_time_drift(time_s, arrays["peak_omega_pe_dt"], fraction),
    }
    if "peak_node_t_e_dense_ev" in arrays:
        drifts["t_e_dense_drift"] = trailing_time_drift(time_s, arrays["peak_node_t_e_dense_ev"], fraction)
    soft = float(block["soft_drift_max"])
    hard = float(block["hard_drift_max"])
    residual_max = float(block["energy_residual_over_electrode_work_max"])
    enforce_after = float(block.get("enforced_after_transit_times", 1.0))
    arming = drift_members_arming(arrays, rule, transit_time_s)
    enforced = transits >= enforce_after if arming is None else bool(arming["armed"])
    windowed = windowed_energy_residual(arrays, block)
    soft_ok = ratio is not None and abs(ratio) < residual_max and all(v is not None and abs(v) < soft for v in drifts.values())
    if windowed is not None and windowed["window_complete"]:
        soft_ok = soft_ok and windowed["ratio"] is not None and windowed["ratio"] < windowed["max"]
    hard_failures = []
    if enforced:
        if windowed is None and (ratio is None or abs(ratio) >= residual_max):
            hard_failures.append(f"energy residual / electrode work {ratio} exceeds {residual_max}")
        for key, value in drifts.items():
            if value is not None and abs(value) >= hard:
                hard_failures.append(f"{key} {value:.3g} exceeds {hard}")
    if windowed is not None and windowed["window_complete"] and windowed["ratio"] is not None and windowed["ratio"] >= windowed["max"]:
        # v2.0.3: enforced from the first complete window, independent of the transit arming of the drift members
        hard_failures.append(
            f"windowed energy residual / electrode work {windowed['ratio']:.3g} over the trailing {windowed['window_steps']} steps "
            f"exceeds {windowed['max']} (finite-grid heating)"
        )
    result = {
        "energy_residual_over_electrode_work": ratio,
        **drifts,
        "soft_ok": bool(soft_ok),
        "enforced": bool(enforced),
        "hard_failures": hard_failures,
        "thresholds": {"energy_residual_over_electrode_work_max": residual_max, "soft_drift_max": soft, "hard_drift_max": hard,
                       "enforced_after_transit_times": enforce_after},
        "window_fraction": fraction,
    }
    if arming is not None:
        # v2.1.1: the drift members' arming state (latch) travels with every record; the legacy transit arming is superseded
        result["drift_members_arming"] = arming
        result["thresholds"]["drift_members_arming"] = {k: arming[k] for k in ("min_transit_times", "settle_quantity", "settle_drift_max", "check_cadence_steps")}
        result["thresholds"]["enforced_after_transit_times_superseded_by_arming_latch"] = True
    if windowed is not None:
        result["windowed_energy_residual_over_electrode_work"] = windowed["ratio"]
        result["windowed_energy_residual_window_steps"] = windowed["window_steps"]
        result["windowed_energy_residual_window_complete"] = windowed["window_complete"]
        result["windowed_energy_residual_electrode_work_j"] = windowed["electrode_work_j"]
        result["windowed_energy_residual_j"] = windowed["residual_j"]
        result["cumulative_residual_is_witness_only"] = True
        result["thresholds"]["windowed_energy_residual_over_electrode_work_max"] = windowed["max"]
        result["thresholds"]["residual_window_steps"] = windowed["window_steps_required"]
    return result


def windowed_energy_residual(arrays: dict[str, np.ndarray], block: Mapping[str, Any]) -> dict[str, Any] | None:
    """v2.0.3: energy-ledger residual over the electrode work, both summed over the records of the trailing window.

    ``block["residual_window_steps"]`` (absent -> None: the v1.4 cumulative member only) is the window in steps;
    the records inside it are those with ``step > last_step - window``; the window is complete when the series
    reaches back at least that far (``first_step <= last_step - window``).  A resume's first record contributes
    zero residual and zero electrode work (the interval ledger restarts there) - a bias of one record in 2000.
    """

    window_steps = block.get("residual_window_steps")
    if window_steps is None or arrays.get("step") is None or arrays["step"].size < 2:
        return None
    window_steps = int(window_steps)
    steps = arrays["step"]
    last = float(steps[-1])
    in_window = steps > last - window_steps
    residual = float(arrays["interval_residual_j"][in_window].sum())
    electrode = float(arrays["interval_electrode_work_j"][in_window].sum())
    # the window's records cover (start, last] where start is the last record OUTSIDE the window (their intervals end at
    # their own step); with every record inside, the coverage starts at the first record (whose own residual is zero)
    outside = steps[~in_window]
    start = float(outside[-1]) if outside.size else float(steps[0])
    return {
        "ratio": residual / electrode if electrode > 0.0 else None,
        "residual_j": residual,
        "electrode_work_j": electrode,
        "window_steps": int(last - start),
        "window_steps_required": window_steps,
        "window_complete": bool(outside.size > 0),
        "max": float(block.get("windowed_energy_residual_over_electrode_work_max", 0.05)),
    }


def evaluate_peak_debye_window(arrays: dict[str, np.ndarray], config: PIC2DConfig) -> dict[str, Any] | None:
    """v2.0.3: the window-mode peak-Debye soft margin as a plateau precondition.

    The gate quantity ``peak_node_window_cells_per_debye`` of the LAST record is already a trailing-window
    (>= 0.6 us) average; ``soft_ok`` holds when it is at or below the declared ``soft_cells_per_debye`` (the
    resolution margin, 2.5 = 20 % under the CIC threshold pi) or when no soft level is declared.  A run whose
    plateau is physical but sits between the soft and the hard level is recorded as such (``soft_ok`` False
    blocks the plateau verdict, the run continues to its budget); the hard level stops it in the simulation.
    Returns None for single-step gates (v1.4 / v2.0.0-v2.0.2) or when the series carries no window sample.
    """

    gate = config.peak_debye_gate
    if gate is None or not gate.windowed or "peak_node_window_cells_per_debye" not in arrays:
        return None
    values = arrays["peak_node_window_cells_per_debye"]
    steps = arrays["peak_node_window_window_steps"]
    last = float(values[-1])
    complete = bool(np.isfinite(steps[-1]) and steps[-1] >= gate.window_steps)
    n_tail = max(values.size // 5, 1)
    tail = values[-n_tail:]
    soft = gate.soft_cells_per_debye
    soft_ok = True if soft is None else bool(complete and np.isfinite(last) and last <= soft)
    return {
        "cells_per_debye_window_last": last if np.isfinite(last) else None,
        "window_steps_last": int(steps[-1]) if np.isfinite(steps[-1]) else None,
        "window_complete_last": complete,
        "trailing_20pct_mean_cells_per_debye_window": float(np.nanmean(tail)) if np.isfinite(tail).any() else None,
        "max_cells_per_debye_window": float(np.nanmax(values)) if np.isfinite(values).any() else None,
        "soft_cells_per_debye": soft,
        "hard_cells_per_debye": gate.max_cells_per_debye,
        "soft_ok": soft_ok,
        "records_above_soft": None if soft is None else int(np.sum(values[np.isfinite(values)] > soft)),
    }


# -- records ----------------------------------------------------------------

def records_to_arrays(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[float]] = {key: [] for key in SERIES_SCALARS + LEDGER_SCALARS + LEDGER_SCALARS_V206 + LEDGER_SCALARS_V23 + LEDGER_SCALARS_V24}
    current_keys = sorted(records[0]["currents_a"]) if records else []
    for key in current_keys:
        arrays[f"current_{key}"] = []
    with_neutral = bool(records) and records[0].get("neutral") is not None
    with_peak = bool(records) and records[0].get("peak_node") is not None
    with_spatial = with_neutral and records[0]["neutral"].get("model") == "neutrals_spatial_v1"    # v2.5.0
    with_meta = with_spatial and records[0]["neutral"].get("metastables") is not None
    if with_neutral:
        for key in NEUTRAL_SCALARS + NEUTRAL_SCALARS_V14 + NEUTRAL_SCALARS_V23:
            arrays[f"neutral_{key}"] = []
        for key in NEUTRAL_LEDGER_KEYS + NEUTRAL_LEDGER_KEYS_V23:
            arrays[f"neutral_ledger_{key}"] = []
    if with_spatial:
        for key in NEUTRAL_SCALARS_V25:
            arrays[f"neutral_{key}"] = []
        for key in NEUTRAL_SPATIAL_LEDGER_KEYS:
            arrays[f"neutral_ledger_{key}"] = []
    if with_meta:
        for key in METASTABLE_SCALARS_V25:
            arrays[f"metastable_{key}"] = []
    if with_peak:
        for key in PEAK_NODE_SCALARS:
            arrays[f"peak_node_{key}"] = []
    with_window = with_peak and records[0]["peak_node"].get("window") is not None   # v2.0.3 window-mode gate
    if with_window:
        for key in PEAK_NODE_WINDOW_SCALARS + PEAK_NODE_WINDOW_WITNESS_SCALARS:
            arrays[f"peak_node_window_{key}"] = []
    with_momentum = bool(records) and records[0].get("momentum") is not None
    with_plume = bool(records) and records[0].get("plume") is not None
    if with_momentum:
        for key in MOMENTUM_SCALARS + MOMENTUM_OPTIONAL_SCALARS + MOMENTUM_OPTIONAL_SCALARS_V23 + MOMENTUM_OPTIONAL_SCALARS_V24:
            arrays[f"momentum_{key}"] = []
    if with_plume:
        for key in PLUME_SCALARS:
            arrays[f"plume_{key}"] = []
    with_see = bool(records) and records[0].get("see") is not None     # v2.2.0: emitting wall
    if with_see:
        for key in SEE_SCALARS:
            arrays[f"see_{key}"] = []
    with_coulomb = bool(records) and records[0].get("coulomb") is not None     # v2.4.0: Coulomb operator on
    if with_coulomb:
        for key in COULOMB_SCALARS:
            arrays[f"coulomb_{key}"] = []
    for record in records:
        for key in SERIES_SCALARS:
            arrays[key].append(float(record[key]))
        for key in LEDGER_SCALARS:
            arrays[key].append(float(record["ledger"][key]))
        for key in LEDGER_SCALARS_V206 + LEDGER_SCALARS_V23 + LEDGER_SCALARS_V24:
            arrays[key].append(float(record["ledger"].get(key, float("nan"))))
        for key in current_keys:
            arrays[f"current_{key}"].append(float(record["currents_a"][key]))
        if with_neutral:
            neutral = record["neutral"]
            for key in NEUTRAL_SCALARS:
                arrays[f"neutral_{key}"].append(float(neutral.get(key, float("nan"))))   # v2.5.0 spatial records lack fixed_point / scale / artificial
            for key in NEUTRAL_SCALARS_V14 + NEUTRAL_SCALARS_V23:
                arrays[f"neutral_{key}"].append(float(neutral.get(key, float("nan"))))
            for key in NEUTRAL_LEDGER_KEYS + NEUTRAL_LEDGER_KEYS_V23:
                arrays[f"neutral_ledger_{key}"].append(float(neutral["ledger"].get(key, 0.0)))  # v1.3 records: no 'recycled'; pre-v2.3.0: no sink
        if with_spatial:
            neutral = record["neutral"]
            for key in NEUTRAL_SCALARS_V25:
                arrays[f"neutral_{key}"].append(float(neutral.get(key, float("nan"))))
            for key in NEUTRAL_SPATIAL_LEDGER_KEYS:
                arrays[f"neutral_ledger_{key}"].append(float(neutral["ledger"].get(key, 0.0)))
        if with_meta:
            meta = record["neutral"].get("metastables") or {}
            for key in METASTABLE_SCALARS_V25:
                arrays[f"metastable_{key}"].append(float(meta.get(key, float("nan"))))
        if with_peak:
            peak = record["peak_node"]
            for key in PEAK_NODE_SCALARS:
                arrays[f"peak_node_{key}"].append(float("nan") if peak[key] is None else float(peak[key]))
        if with_window:
            window = record["peak_node"].get("window") or {}
            for key in PEAK_NODE_WINDOW_SCALARS:
                value = window.get(key)
                arrays[f"peak_node_window_{key}"].append(float("nan") if value is None else float(value))
            witness = (window.get("occupancy_floor_peak") or {}).get("cells_per_debye")     # v2.0.6 witness
            arrays["peak_node_window_occupancy_floor_cells_per_debye"].append(float("nan") if witness is None else float(witness))
        if with_momentum:
            momentum = record["momentum"]
            for key in MOMENTUM_SCALARS:
                arrays[f"momentum_{key}"].append(float(momentum[key]))
            for key in MOMENTUM_OPTIONAL_SCALARS + MOMENTUM_OPTIONAL_SCALARS_V23 + MOMENTUM_OPTIONAL_SCALARS_V24:   # cathode / ion MCC / Coulomb only
                value = momentum.get(key)
                arrays[f"momentum_{key}"].append(float("nan") if value is None else float(value))
        if with_plume:
            plume = record["plume"]
            for key in PLUME_SCALARS:
                value = plume.get(key)
                arrays[f"plume_{key}"].append(float("nan") if value is None else float(value))
        if with_see:
            see = record.get("see") or {}
            for key in SEE_SCALARS:
                value = see.get(key)
                arrays[f"see_{key}"].append(float("nan") if value is None else float(value))
        if with_coulomb:
            coulomb = record.get("coulomb") or {}
            for key in COULOMB_SCALARS:
                value = coulomb.get(key)
                arrays[f"coulomb_{key}"].append(float("nan") if value is None else float(value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in arrays.items()}


def status_from_record(
    record: dict[str, Any], config: PIC2DConfig, plasma_volume_m3: float, *, wall_seconds_total: float,
    ms_per_step: float | None, plateau: dict[str, Any] | None, triad: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_e = record["electrons"] * config.macro_weight
    omega = record["peak_omega_pe_dt"] / config.dt_s
    peak_node = omega * omega * EPSILON_0 * ELECTRON_MASS_KG / ELEMENTARY_CHARGE_C**2
    t_e = (2.0 / 3.0) * record["kinetic_electron_j"] / (max(n_e, 1.0) * ELEMENTARY_CHARGE_C)
    line = {
        "step": record["step"],
        "time_s": record["time_s"],
        "electrons": record["electrons"],
        "ions": record["ions"],
        "discharge_a": record["currents_a"]["discharge_a"],
        "exit_ion_beam_a": record["currents_a"]["exit_ion_beam_a"],
        "ionization_rate_per_s": record["currents_a"]["ionization_rate_per_s"],
        "n_e_peak_node_per_m3": peak_node,
        "n_e_mean_per_m3": n_e / plasma_volume_m3,
        "t_e_mean_ev": t_e,
        "omega_pe_dt_max": record["peak_omega_pe_dt"],
        "phi_max_v": record["phi_max_v"],
        "wall_seconds_total": wall_seconds_total,
        "ms_per_step": ms_per_step,
        "plateau": None if plateau is None else {
            key: plateau[key] for key in ("reached", "transit_times_elapsed", *plateau["tracked"])
        },
    }
    neutral = record.get("neutral")
    if neutral is not None:
        line["n_g_per_m3"] = neutral["density_per_m3"]
        line["n_g_fixed_point_per_m3"] = neutral.get("fixed_point_per_m3")     # None for the v2.5.0 spatial model
        line["effusion_rate_per_s"] = neutral["effusion_rate_per_s"]
        line["neutral_ledger_residual_atoms"] = neutral["interval_ledger_residual_atoms"]
        if "recycled_rate_per_s" in neutral:  # v1.4
            line["recycled_rate_per_s"] = neutral["recycled_rate_per_s"]
            line["gross_utilisation"] = neutral["gross_utilisation"]
            line["net_utilisation"] = neutral["net_utilisation"]
        if neutral.get("model") == "neutrals_spatial_v1":   # v2.5.0
            line["neutral_model"] = neutral["model"]
            for key in ("axis_density_anode_per_m3", "axis_density_exit_per_m3", "macro_neutrals", "macro_metastables", "debt_ground_atoms"):
                line[key] = neutral[key]
            if neutral.get("metastables") is not None:
                line["metastable_fraction_of_ground"] = neutral["metastables"]["fraction_of_ground"]
                line["stepwise_fraction_of_ionization"] = neutral["metastables"]["stepwise_fraction_of_ionization"]
    peak = record.get("peak_node")
    if peak is not None:  # v1.4 peak-node Debye sample
        line["peak_node"] = {key: peak[key] for key in ("n_e_peak_per_m3", "t_e_peak_ev", "cells_per_debye", "macro_particles_at_peak",
                                                        "t_e_dense_ev", "z_m", "r_m")}
        if "gate_enforced" in peak:
            line["peak_node"]["gate_enforced"] = peak["gate_enforced"]
            line["peak_node"]["gate_max_cells_per_debye"] = peak["gate_max_cells_per_debye"]
        if peak.get("window") is not None:  # v2.0.3 window-mode gate quantity
            window = peak["window"]
            line["peak_node"]["gate_mode"] = peak.get("gate_mode")
            line["peak_node"]["window"] = {key: window[key] for key in ("cells_per_debye", "n_e_peak_per_m3", "t_e_peak_ev", "node",
                                                                        "window_steps", "window_complete", "gate_enforced", "soft_exceeded",
                                                                        "mean_macro_particles_at_peak", "resolved_nodes")}
            if "occupancy_floor_peak" in window:   # v2.0.6 accumulated floor: the gated node's accumulation + the occupancy-floor witness
                line["peak_node"]["window"]["min_accumulated_macro_particle_steps_at_peak"] = window["min_accumulated_macro_particle_steps_at_peak"]
                line["peak_node"]["window"]["accumulated_macro_particle_steps_at_peak"] = window["accumulated_macro_particle_steps_at_peak"]
                line["peak_node"]["window"]["occupancy_floor_peak"] = {
                    key: window["occupancy_floor_peak"][key] for key in ("cells_per_debye", "node", "resolved_nodes", "mean_macro_particles_at_peak")}
    if triad is not None:
        line["grid_heating_triad"] = {key: triad[key] for key in triad if key not in ("thresholds", "window_fraction")}
    momentum = record.get("momentum")
    if momentum is not None:  # v2.0 thrust ledger (interval values; the window averages are in the summary)
        line["thrust"] = {key: momentum[key] for key in ("thrust_flux_n", "cold_gas_thrust_n", "thrust_total_n", "thrust_balance_n",
                                                       "closure_fraction", "electrostatic_force_thruster_n", "interval_ledger_residual_kg_m_s")}
        if "cathode_emission_next_a" in momentum:
            line["thrust"]["cathode_emission_next_a"] = momentum["cathode_emission_next_a"]
        if "cathode_emission_a" in record["currents_a"]:
            line["cathode_emission_a"] = record["currents_a"]["cathode_emission_a"]
    plume = record.get("plume")
    if plume is not None:  # v2.0 plume-boundary sample
        line["plume"] = {key: plume[key] for key in ("charge_fraction_of_peak", "far_field_phi_max_abs_deviation_v",
                                                   "exit_plane_axis_potential_v", "acceleration_z90_m", "acceleration_z10_m")}
        for key in ("charge_fraction_of_peak_raw", "far_field_resolved_nodes",                        # v2.0.1
                    "far_field_window_steps", "far_field_window_complete", "charge_fraction_of_peak_window_raw"):   # v2.0.2
            if key in plume:
                line["plume"][key] = plume[key]
        for key in ("gate_enforced", "gate_armed"):
            if key in plume:
                line["plume"][key] = plume[key]
    return line


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    # append-only logs (not canonical artifacts): a NaN in a diagnostic must not end a 12 h run
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


# -- checkpointing ----------------------------------------------------------

CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_NAME = "checkpoint-latest"


TERMINAL_STATE_KEYS = ("finished", "stop_reason", "finalized_from_step", "finalization_recovery", "finalization_error")


def _demote_terminal_state(run_state: dict[str, Any], *, event: str, step: int, utc: str, summary_present: bool) -> dict[str, Any] | None:
    """Move a previous session's terminal block into ``run_state["history"]`` and reset the live flags.

    v2.1 (plume attempt 8 lesson): a resume used to leave ``finished: true``,
    ``stop_reason: wall_clock_budget_reached``, ``finalized_from_step`` and the
    ``finalization_recovery`` block of the session it continued in ``run_state.json``
    while stepping - only the advancing ``checkpoint_step`` and the new session entry
    proved the run was live.  The information is kept (one ``history`` entry per
    demotion, with the event, the step and the time it happened and whether a
    ``summary.json`` of the superseded stop was on disk), the live keys are reset:
    ``finished`` False, the other terminal keys removed.  Returns the history entry
    (None when there was nothing to demote).
    """

    present = {key: run_state[key] for key in TERMINAL_STATE_KEYS if key in run_state}
    if not present or (set(present) == {"finished"} and not present["finished"]):
        run_state["finished"] = False
        return None
    entry: dict[str, Any] = {"event": event, "utc": utc, "step": int(step), "superseded_summary_json_on_disk": bool(summary_present)} | present
    run_state.setdefault("history", []).append(entry)
    for key in TERMINAL_STATE_KEYS:
        run_state.pop(key, None)
    run_state["finished"] = False
    return entry


def save_checkpoint_atomic(results: Path, sim: Simulation, config: PIC2DConfig, field_map: MagneticFieldMap, xs_sha256: str | None) -> Path:
    """Write the checkpoint into a fresh directory, then swap it in (old copy kept until the swap is done).

    The field map is passed whole so the checkpoint binds its platform-independent source identity and keeps an
    anchor copy of the node arrays (cross-platform resume policy, see ``artifacts.save_checkpoint``).
    """

    tmp = results / f"{CHECKPOINT_DIR}-tmp"
    old = results / f"{CHECKPOINT_DIR}-old"
    live = results / CHECKPOINT_DIR
    for stale in (tmp, old):
        if stale.exists():
            shutil.rmtree(stale)
    tmp.mkdir(parents=True)
    artifacts.save_checkpoint(tmp, CHECKPOINT_NAME, sim.state, config, field_sha256=field_map.sha256, field=field_map,
                              cross_section_sha256=xs_sha256, backend=sim.backend.name)
    if live.exists():
        live.rename(old)
    tmp.rename(live)
    if old.exists():
        shutil.rmtree(old)
    return live / f"{CHECKPOINT_NAME}.json"


def find_checkpoint(results: Path) -> Path | None:
    for name in (CHECKPOINT_DIR, f"{CHECKPOINT_DIR}-old"):
        candidate = results / name / f"{CHECKPOINT_NAME}.json"
        if candidate.is_file():
            return candidate
    return None


# -- shared setup -------------------------------------------------------------

def plume_extension_path(protocol: Mapping[str, Any] | None) -> Path | None:
    """v2.1: ``protocol["field_plume_extension"]`` (repository-relative) names the plume field-extension declaration;
    absent = the v2.0 default (spec/pic2d/p2-field-plume-extension-v1.json, the authority's checkpoint)."""

    if protocol is None or protocol.get("field_plume_extension") is None:
        return None
    path = REPOSITORY_ROOT / str(protocol["field_plume_extension"])
    if not path.is_file():
        raise PIC2DValidationError(f"field_plume_extension {protocol['field_plume_extension']!r} is not a file")
    return path


def load_inputs(config: PIC2DConfig, field_map: MagneticFieldMap | None, cross_sections: XenonCrossSections | None, *,
                protocol: Mapping[str, Any] | None = None):
    if field_map is None:
        if config.grid.geometry.has_plume:
            # v2.0: direct P2 node evaluation over the L-shaped box (spec/pic2d/p2-field-plume-extension-v1.json);
            # v2.1: the protocol may name an extension file with its own (larger-box) checkpoint declaration
            field_map = p2_plume_field_map(REPOSITORY_ROOT, config.grid, role="primary", extension_path=plume_extension_path(protocol))
        else:
            psi_field, evidence = build_p2_psi_field(REPOSITORY_ROOT, role="primary")
            field_map = sample_field_map(psi_field, config.grid, evidence)
    if cross_sections is None and config.mcc is not None:
        if config.mcc.collision_set is not None:
            cross_sections = config.mcc.collision_set.load_electron_cross_sections()   # v2.3.0: the declared, hash-checked set
        else:
            cross_sections = XenonCrossSections.from_file()
    return field_map, cross_sections


def ledger_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if arrays.get("step") is None or arrays["step"].size < 2:
        return {}
    residual = arrays["interval_residual_j"][1:]
    electrode = arrays["interval_electrode_work_j"][1:]
    sources = arrays["interval_sources_j"][1:]
    total = arrays["total_energy_j"]
    return {
        "cumulative_residual_j": float(residual.sum()),
        "cumulative_electrode_work_j": float(electrode.sum()),
        "total_energy_change_j": float(total[-1] - total[0]),
        "gross_source_turnover_j": float(np.sum(np.abs(sources))),
        "cumulative_residual_over_electrode_work": float(residual.sum() / electrode.sum()) if abs(electrode.sum()) > 0 else None,
        "interval_residual_rms_j": float(np.sqrt(np.mean(residual**2))),
        "note": "intervals restart at every resume (the first record of a session has zero residual and electrode work)",
    }


def spatial_neutral_summary(arrays: dict[str, np.ndarray], sim: Simulation) -> dict[str, Any] | None:
    """v2.5.0: summary of the spatial neutral model (ledger closure, trailing profile readings, metastable fraction)."""

    if not getattr(sim, "spatial_neutrals_on", False) or "neutral_atoms_ground" not in arrays:
        return None
    spatial = sim.backend.spatial
    tail = max(arrays["neutral_density_per_m3"].size // 5, 1)
    ledger = {key: float(arrays[f"neutral_ledger_{key}"][-1]) for key in NEUTRAL_SPATIAL_LEDGER_KEYS}
    f_acc = float(arrays["neutral_time_acceleration"][-1])
    out: dict[str, Any] = {
        **spatial.to_dict(),
        "final_channel_mean_density_per_m3": float(arrays["neutral_density_per_m3"][-1]),
        "final_axis_density_anode_per_m3": float(arrays["neutral_axis_density_anode_per_m3"][-1]),
        "final_axis_density_exit_per_m3": float(arrays["neutral_axis_density_exit_per_m3"][-1]),
        "final_atoms_ground": float(arrays["neutral_atoms_ground"][-1]),
        "final_atoms_metastable": float(arrays["neutral_atoms_metastable"][-1]),
        "final_macro_neutrals": float(arrays["neutral_macro_neutrals"][-1]),
        "final_macro_metastables": float(arrays["neutral_macro_metastables"][-1]),
        "neutral_time_s_total": float(arrays["neutral_neutral_time_s"][-1]),
        "trailing_20pct_mean_density_per_m3": float(np.mean(arrays["neutral_density_per_m3"][-tail:])),
        "trailing_20pct_mean_ionization_rate_per_s": float(np.mean(arrays["neutral_ionization_rate_per_s"][-tail:])),
        "trailing_20pct_mean_effusion_rate_per_s": float(np.mean(arrays["neutral_effusion_rate_per_s"][-tail:])),
        "trailing_20pct_mean_recycled_rate_per_s": float(np.mean(arrays["neutral_recycled_rate_per_s"][-tail:])),
        "trailing_20pct_mean_neutral_exit_thrust_n": float(np.mean(arrays["neutral_neutral_exit_thrust_n"][-tail:])),
        "gross_utilisation_trailing": float(np.mean(arrays["neutral_gross_utilisation"][-tail:])),
        "propellant_utilisation_trailing": float(np.mean(arrays["neutral_gross_utilisation"][-tail:])),    # the v4 assess key (gross)
        "net_utilisation_trailing": float(np.mean(arrays["neutral_net_utilisation"][-tail:])),
        "cumulative_ledger_atoms_neutral_time": ledger,
        "cumulative_ledger_atoms_real_time_plasma_terms": {
            key: ledger[key] / f_acc for key in ("neutral_ionized", "neutral_cex_converted", "neutral_excited_to_pool", "neutral_recycled", "neutral_fast_in",
                                                  "meta_ionized", "meta_superelastic")
        },
        "max_interval_ledger_residual_atoms": float(np.max(np.abs(arrays["neutral_interval_ledger_residual_atoms"]))),
        "max_interval_meta_ledger_residual_atoms": float(np.max(np.abs(arrays["neutral_interval_meta_ledger_residual_atoms"]))),
        "max_sink_consistency_atoms": float(np.max(np.abs(arrays["neutral_sink_consistency_atoms"]))),
        "final_debt_ground_atoms": float(arrays["neutral_debt_ground_atoms"][-1]),
        "note": ("neutral time = time_acceleration x plasma time; the ledger identities close on the true counts (particles + carries - debts); "
                 "only the quasi-steady profile is physical when time_acceleration > 1"),
    }
    if "metastable_fraction_of_ground" in arrays:
        out["metastables"] = {
            "final_fraction_of_ground": float(arrays["metastable_fraction_of_ground"][-1]),
            "trailing_20pct_mean_fraction_of_ground": float(np.mean(arrays["metastable_fraction_of_ground"][-tail:])),
            "trailing_20pct_mean_stepwise_fraction_of_ionization": float(np.mean(arrays["metastable_stepwise_fraction_of_ionization"][-tail:])),
            "trailing_20pct_mean_production_rate_per_s": float(np.mean(arrays["metastable_production_rate_per_s"][-tail:])),
            "trailing_20pct_mean_stepwise_ionization_rate_per_s": float(np.mean(arrays["metastable_stepwise_ionization_rate_per_s"][-tail:])),
            "trailing_20pct_mean_superelastic_rate_per_s": float(np.mean(arrays["metastable_superelastic_rate_per_s"][-tail:])),
            "trailing_20pct_mean_wall_deexcitation_rate_per_s": float(np.mean(arrays["metastable_wall_deexcitation_rate_per_s"][-tail:])),
        }
    return out


def neutral_summary(arrays: dict[str, np.ndarray], sim: Simulation, initial_density: float) -> dict[str, Any] | None:
    if getattr(sim, "spatial_neutrals_on", False):
        return spatial_neutral_summary(arrays, sim)
    if sim.neutrals is None or "neutral_density_per_m3" not in arrays:
        return None
    inventory = sim.neutrals
    ledger = {key: float(arrays[f"neutral_ledger_{key}"][-1]) for key in NEUTRAL_LEDGER_KEYS}
    # v1.4 balance: fed + recycled - ionised - effused - artificial = V (n_1 - n_0); 'recycled' is 0 for v1.3 runs
    closure = ledger["fed"] + ledger["recycled"] - ledger["ionized"] - ledger["effused"] - ledger["artificial"]
    n_final = float(arrays["neutral_density_per_m3"][-1])
    expected = inventory.volume_m3 * (n_final - initial_density)
    tail = arrays["neutral_density_per_m3"][-max(arrays["neutral_density_per_m3"].size // 5, 1):]
    s_tail = arrays["neutral_ionization_rate_per_s"][-tail.size:]
    recycled_tail = arrays["neutral_recycled_rate_per_s"][-tail.size:]
    recycled_mean = float(np.nanmean(recycled_tail)) if np.any(np.isfinite(recycled_tail)) else 0.0
    feed = inventory.config.feed_atoms_per_s
    relaxation_on = getattr(inventory, "relaxation_on", inventory.config.relaxation_time_s is not None)
    return {
        **inventory.to_dict(),
        "initial_density_per_m3": initial_density,
        "final_density_per_m3": n_final,
        "final_fixed_point_per_m3": float(arrays["neutral_fixed_point_per_m3"][-1]),
        "max_density_over_zero_ionization": float(np.max(arrays["neutral_density_per_m3"])) / inventory.zero_ionization_density,
        "trailing_20pct_mean_density_per_m3": float(np.mean(tail)),
        "trailing_20pct_mean_ionization_rate_per_s": float(np.mean(s_tail)),
        "trailing_20pct_mean_recycled_rate_per_s": recycled_mean,
        "trailing_20pct_analytic_fixed_point_per_m3": inventory.fixed_point(float(np.mean(s_tail)), recycled_mean),
        "trailing_20pct_mean_artificial_rate_per_s": float(np.mean(arrays["neutral_artificial_rate_per_s"][-tail.size:])),
        "trailing_20pct_artificial_rate_rms_per_s": float(np.sqrt(np.mean(arrays["neutral_artificial_rate_per_s"][-tail.size:] ** 2))),
        "cumulative_ledger_atoms": ledger,
        "cumulative_ledger_closure_atoms": closure - expected,
        "cumulative_ledger_closure_relative_to_inventory": (closure - expected) / (inventory.volume_m3 * initial_density),
        "max_interval_ledger_residual_atoms": float(np.max(np.abs(arrays["neutral_interval_ledger_residual_atoms"]))),
        "propellant_utilisation_trailing": float(np.mean(s_tail)) / feed,
        "gross_utilisation_trailing": float(np.mean(s_tail)) / feed,
        "net_utilisation_trailing": (float(np.mean(s_tail)) - recycled_mean) / feed,
        "utilisation_note": "gross = S / Q_in (ionisations per fed atom); net = (S - recycled wall/anode ions) / Q_in = beam ions per fed "
                            "atom at the fixed point; without wall recycling (v1.3) the two coincide and the gross value overstates the "
                            "atoms consumed",
        "note": ("the transient toward the fixed point is artificial (relaxation_time_s); only the fixed point is physical"
                 if relaxation_on else "no artificial relaxation: n_g evolves on the physical effusion time scale V/c"),
    }


XENON_MASS_KG = 2.1801714e-25
G0_M_PER_S2 = 9.80665


def plume_summary(
    arrays: dict[str, np.ndarray], maps: dict[str, np.ndarray], window_range: tuple[int, int], config: PIC2DConfig,
    window_currents: Mapping[str, float | None],
) -> dict[str, Any] | None:
    """v2.0 plume block: window-averaged thrust with the closure check, divergence, IEDF, performance numbers.

    Development numbers (claim boundary): the thrust is the axial momentum flux through the far-field
    boundary (+ the cold-gas effusion of the inventory) averaged over the reporting window; the
    momentum-balance thrust ``-F_on_thruster`` from the particle ledger and the Maxwell-stress force on
    the solid boundaries are the independent checks; specific impulse and anode efficiency follow from
    the declared feed and the window discharge current.
    """

    if "momentum_thrust_flux_n" not in arrays or not config.grid.geometry.has_plume:
        return None
    steps = arrays["step"]
    in_window = (steps > window_range[0]) & (steps <= window_range[1])
    if not in_window.any():
        in_window = steps >= steps[max(steps.size - max(steps.size // 5, 1), 0)]
    mean = lambda key: float(np.nanmean(arrays[key][in_window]))  # noqa: E731
    thrust_flux = mean("momentum_thrust_flux_n")
    cold_gas = mean("momentum_cold_gas_thrust_n")
    thrust_total = thrust_flux + cold_gas
    balance = mean("momentum_thrust_balance_n")
    scale = max(abs(thrust_flux), abs(balance))
    closure = (thrust_flux - balance) / scale if scale > 0.0 else None   # window means; normalised by the larger of the two
    # divergence: half-angle containing 95 % of the window's far-field ion crossings (about the aperture centre)
    counts = np.asarray(maps.get("plume_ion_counts_per_theta", np.zeros(0)), dtype=np.float64)
    edges = np.asarray(maps.get("plume_theta_edges_deg", np.zeros(0)), dtype=np.float64)
    half_angle = None
    if counts.size and counts.sum() > 0:
        cumulative = np.cumsum(counts) / counts.sum()
        half_angle = float(edges[1:][int(np.searchsorted(cumulative, 0.95))])
    iedf = np.asarray(maps.get("iedf_ion_counts", np.zeros(0)), dtype=np.float64)
    e_edges = np.asarray(maps.get("iedf_edges_ev", np.zeros(0)), dtype=np.float64)
    mean_energy = peak_energy = None
    if iedf.size and iedf.sum() > 0:
        centres = 0.5 * (e_edges[:-1] + e_edges[1:])
        mean_energy = float(np.sum(centres * iedf) / iedf.sum())
        peak_energy = float(centres[int(np.argmax(iedf))])
    feed_block = config.neutral_inventory if config.neutral_inventory is not None else config.neutrals_spatial   # v2.5.0: either neutral model
    feed_kg_per_s = None if feed_block is None else feed_block.feed_atoms_per_s * XENON_MASS_KG
    discharge = window_currents.get("discharge_a")
    power_w = None if discharge is None else float(config.potentials.anode_v) * float(discharge)
    isp = None if feed_kg_per_s in (None, 0.0) else thrust_total / (feed_kg_per_s * G0_M_PER_S2)
    efficiency = None
    if feed_kg_per_s not in (None, 0.0) and power_w not in (None, 0.0):
        efficiency = thrust_total**2 / (2.0 * feed_kg_per_s * power_w)
    return {
        "window_step_range": list(window_range),
        "window_samples": int(in_window.sum()),
        "thrust_flux_n": thrust_flux,
        "thrust_flux_ions_n": mean("momentum_beam_momentum_rate_ions_n"),
        "thrust_flux_electrons_n": mean("momentum_beam_momentum_rate_electrons_n"),
        "cathode_injected_momentum_rate_n": mean("momentum_injected_momentum_rate_n"),
        "cold_gas_thrust_n": cold_gas,
        "thrust_total_n": thrust_total,
        "thrust_balance_n": balance,
        "closure_fraction": closure,
        "electrostatic_force_thruster_n": mean("momentum_electrostatic_force_thruster_n"),
        "electrostatic_force_far_field_n": mean("momentum_electrostatic_force_far_field_n"),
        "absorbed_momentum_rate_n": mean("momentum_absorbed_momentum_rate_n"),
        "field_impulse_rate_n": mean("momentum_field_impulse_rate_n"),
        "stored_momentum_rate_n": mean("momentum_dp_rate_n"),
        "collision_momentum_rate_n": mean("momentum_collision_momentum_rate_n"),
        "ledger_residual_max_kg_m_s": float(np.nanmax(np.abs(arrays["momentum_interval_ledger_residual_kg_m_s"]))),
        "divergence_half_angle_95_deg": half_angle,
        "far_field_ion_crossings_in_window": float(counts.sum()) if counts.size else None,
        "iedf_mean_energy_ev": mean_energy,
        "iedf_peak_energy_ev": peak_energy,
        "iedf_peak_minus_anode_v": None if peak_energy is None else peak_energy - float(config.potentials.anode_v),
        "exit_plane_axis_potential_v": mean("plume_exit_plane_axis_potential_v") if "plume_exit_plane_axis_potential_v" in arrays else None,
        "acceleration_z90_m": mean("plume_acceleration_z90_m") if "plume_acceleration_z90_m" in arrays else None,
        "acceleration_z10_m": mean("plume_acceleration_z10_m") if "plume_acceleration_z10_m" in arrays else None,
        "acceleration_width_m": mean("plume_acceleration_width_m") if "plume_acceleration_width_m" in arrays else None,
        "charge_fraction_of_peak_max": float(np.nanmax(arrays["plume_charge_fraction_of_peak"])) if "plume_charge_fraction_of_peak" in arrays else None,
        "charge_fraction_of_peak_raw_max": (float(np.nanmax(arrays["plume_charge_fraction_of_peak_raw"]))
                                            if "plume_charge_fraction_of_peak_raw" in arrays and np.isfinite(arrays["plume_charge_fraction_of_peak_raw"]).any() else None),
        # v2.0.2: the unrestricted trailing-window statistic, the window length and the resolved-node count at the last record
        "charge_fraction_of_peak_window_raw_max": (float(np.nanmax(arrays["plume_charge_fraction_of_peak_window_raw"]))
                                                   if "plume_charge_fraction_of_peak_window_raw" in arrays
                                                   and np.isfinite(arrays["plume_charge_fraction_of_peak_window_raw"]).any() else None),
        "far_field_window_steps_final": (int(arrays["plume_far_field_window_steps"][-1])
                                         if "plume_far_field_window_steps" in arrays and np.isfinite(arrays["plume_far_field_window_steps"][-1]) else None),
        "far_field_resolved_nodes_final": (int(arrays["plume_far_field_resolved_nodes"][-1])
                                           if "plume_far_field_resolved_nodes" in arrays and np.isfinite(arrays["plume_far_field_resolved_nodes"][-1]) else None),
        "mass_flow_kg_per_s": feed_kg_per_s,
        "discharge_power_w": power_w,
        "specific_impulse_s": isp,
        "anode_efficiency": efficiency,
        "claim_boundary": (
            "development numbers: window-averaged momentum flux through a 12 mm plume box with a Dirichlet far field; "
            "closure = (T_flux - (-F_on_thruster)) / max(|T_flux|, |F|) of the window means from the particle ledger (the stored-"
            "momentum rate, collisions and the far-field electrostatic force are the reported non-closing terms), the Maxwell-stress force on the solid "
            "boundaries is the independent field-side check; Isp and anode efficiency from the declared feed and the window "
            "discharge current; not a performance prediction"
        ),
    }


def write_final_artifacts(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    results: Path,
    sim: Simulation,
    config: PIC2DConfig,
    field_map: MagneticFieldMap,
    xs_sha: str | None,
    records: list[dict[str, Any]],
    maps: dict[str, np.ndarray],
    window_range: tuple[int, int],
    maps_kind: str,
    stop_reason: str,
    gate_error: str | None,
    run_state: dict[str, Any],
    session: dict[str, Any],
    setup_seconds: float,
    wall_session: float,
    gpu_samples: list[float | None],
    gpu_sampler: dict[str, Any] | None = None,
) -> Path:
    state = sim.state
    budget = protocol_budget(protocol)
    rule = protocol["stopping_rule"]
    transit_time = float(budget["ion_transit_time_s"])
    arrays = records_to_arrays(records) if records else {}
    plateau = None
    triad = None
    peak_node_summary = None
    if records:
        plateau = evaluate_plateau(arrays["time_s"], arrays["current_discharge_a"], arrays["electrons"], rule, transit_time,
                                   arrays.get("neutral_density_per_m3"))
        triad = evaluate_triad(arrays, rule, transit_time)
        if triad is not None:
            plateau["triad_soft_ok"] = triad["soft_ok"]
            plateau["reached"] = bool(plateau["reached"] and triad["soft_ok"])
        debye_window = evaluate_peak_debye_window(arrays, config)
        if debye_window is not None:   # v2.0.3: the soft resolution margin is a plateau precondition
            plateau["peak_debye_soft_ok"] = debye_window["soft_ok"]
            plateau["reached"] = bool(plateau["reached"] and debye_window["soft_ok"])
        if "peak_node_cells_per_debye" in arrays:
            n_tail = max(arrays["step"].size // 5, 1)
            gate = config.peak_debye_gate
            peak_node_summary = {
                "window": debye_window,
                "gate_mode": "window" if (gate is not None and gate.windowed) else ("single_step" if gate is not None else None),
                "trailing_20pct_mean_cells_per_debye": float(np.mean(arrays["peak_node_cells_per_debye"][-n_tail:])),
                "max_cells_per_debye": float(np.max(arrays["peak_node_cells_per_debye"])),
                "trailing_20pct_mean_n_e_peak_per_m3": float(np.mean(arrays["peak_node_n_e_peak_per_m3"][-n_tail:])),
                "trailing_20pct_mean_t_e_peak_ev": float(np.mean(arrays["peak_node_t_e_peak_ev"][-n_tail:])),
                "trailing_20pct_mean_t_e_dense_ev": float(np.mean(arrays["peak_node_t_e_dense_ev"][-n_tail:])),
                "trailing_20pct_mean_macro_particles_at_peak": float(np.mean(arrays["peak_node_macro_particles_at_peak"][-n_tail:])),
                "trailing_20pct_mean_peak_z_m": float(np.mean(arrays["peak_node_z_m"][-n_tail:])),
                "gate": None if gate is None else gate.to_dict(),
                "note": "peak node = densest node holding >= min_macro_particles_at_peak macro-electrons; the gate fails closed on "
                        "max(dr, dz) / lambda_D there (review blocker 1: resolve the peak, not the mean); v2.0.3 window mode: the gated "
                        "statistic is the trailing-window (interval-averaged) peak in 'window', the single-step sample is the witness; "
                        "v2.0.6 (gate.min_accumulated_macro_particle_steps_at_peak set): the resolved set is the nodes with that many "
                        "accumulated macro-electron-steps over the window and the v2.0.3 occupancy-floor peak is the witness "
                        "(series peak_node_window_occupancy_floor_cells_per_debye)",
            }
    maps_sha = artifacts.write_npz(results / "maps.npz", maps)
    series_sha = artifacts.write_npz(results / "series.npz", arrays) if arrays else None
    checkpoint_json, checkpoint_npz = artifacts.save_checkpoint(
        results, "checkpoint-final", state, config, field_sha256=field_map.sha256, field=field_map, cross_section_sha256=xs_sha,
        backend=sim.backend.name,
    )
    window = int(maps["window_steps"][0])
    plasma = sim.masks.plasma_node
    window_currents: dict[str, float | None] = {}
    if arrays:
        steps_arr = arrays["step"]
        in_window = (steps_arr > window_range[0]) & (steps_arr <= window_range[1])
        window_currents = {
            key[len("current_"):]: float(np.mean(arrays[key][in_window])) if in_window.any() else None
            for key in arrays if key.startswith("current_")
        }

    def stat(name: str, fn: Callable[[np.ndarray], float]) -> float | None:
        return _finite(fn(maps[name][plasma])) if window else None

    status_path = results / "status.jsonl"
    summary = {
        "schema_version": "cft-revival.pic2d-cft-steady-state.summary/0.2.0",
        "experiment_id": protocol["experiment_id"],
        "model_version": protocol["model_version"],
        "status": protocol["status"],
        "classification": protocol["classification"],
        "case": protocol["case"],
        "protocol_sha256": _file_sha256(protocol_path) if protocol_path.is_file() else None,
        "git_head": git_head(),
        "backend": sim.backend.name,
        "steps_completed": int(state.step),
        "simulated_time_s": float(state.time_s),
        "ion_transit_times": float(state.time_s) / transit_time,
        "stop_reason": stop_reason,
        "stability_gate_message": gate_error,
        "plateau": plateau,
        "grid_heating_triad": triad,
        "peak_node_debye": peak_node_summary,
        "sessions": run_state["sessions"],
        "wall_seconds_total": run_state["wall_seconds_total"],
        # v2.0: ignition gate evaluation and the cathode field-line connectivity check (None for v1.x protocols)
        "ignition": run_state.get("ignition", evaluate_ignition(arrays, protocol["stopping_rule"]) if arrays else None),
        "cathode_connectivity": run_state.get("cathode_connectivity"),
        "wall_seconds_setup_this_session": setup_seconds,
        "ms_per_step_this_session": (
            1e3 * wall_session / max(state.step - session["resumed_from_step"], 1) if state.step > session["resumed_from_step"] else None
        ),
        # nvidia-smi failures / timeouts are None (attempt-7 lesson: a NaN sample made the summary non-canonical);
        # v2.0.2: sampled on a background thread at gpu_utilisation_sampler.interval_seconds (None for older runs)
        "gpu_utilisation_percent_samples": [None if sample is None else _finite(sample) for sample in gpu_samples],
        "gpu_utilisation_sampler": gpu_sampler,
        "maps_kind": maps_kind,
        "averaging_window_steps": window,
        "averaging_window_step_range": list(window_range),
        "final_counts": {"electrons": state.electrons.count, "ions": state.ions.count},
        "peak_counts": {"electrons": int(arrays["electrons"].max()), "ions": int(arrays["ions"].max())} if arrays else None,
        "final_series": records[-1] if records else None,
        "window_currents_a": window_currents,
        "ledger": ledger_summary(arrays),
        "neutral_inventory": neutral_summary(arrays, sim, float(sim.neutrals.initial_density) if sim.neutrals is not None else float(config.mcc.neutral_density_per_m3)) if config.mcc is not None else None,
        "plume": plume_summary(arrays, maps, window_range, config, window_currents) if arrays else None,
        "window_maps_summary": {
            "n_e_peak_per_m3": stat("n_e_per_m3", np.nanmax),
            "n_e_mean_per_m3": stat("n_e_per_m3", np.nanmean),
            "n_i_peak_per_m3": stat("n_i_per_m3", np.nanmax),
            "phi_min_v": stat("phi_v", np.nanmin),
            "phi_max_v": stat("phi_v", np.nanmax),
            "t_e_max_ev": stat("t_e_ev", np.nanmax),
            "t_e_density_weighted_mean_ev": (
                _finite(np.nansum(maps["t_e_ev"][plasma] * maps["n_e_per_m3"][plasma]) / max(np.nansum(maps["n_e_per_m3"][plasma]), 1e-300))
                if window else None
            ),
            "ionization_rate_peak_per_m3_s": stat("ionization_rate_per_m3_s", np.nanmax),
            "wall_ion_flux_peak_per_m2_s": _finite(np.nanmax(maps["wall_ion_flux_per_m2_s"])) if window else None,
            "exit_ion_current_a": _finite(np.sum(maps["exit_ion_current_density_a_per_m2"] * _exit_areas(config.grid))) if window else None,
        },
        "budget_check": {
            "n_e_peak_over_n_max": (stat("n_e_per_m3", np.nanmax) or 0.0) / float(budget["n_max_per_m3"]) if window else None,
            "n_e_mean_over_projected_n_eq": (stat("n_e_per_m3", np.nanmean) or 0.0) / float(budget["n_eq_projected_per_m3"]) if window else None,
            "max_observed_omega_pe_dt": float(arrays["peak_omega_pe_dt"].max()) if arrays else None,
        },
        "artifacts": {
            "maps_npz_sha256": maps_sha,
            "series_npz_sha256": series_sha,
            "checkpoint_json": checkpoint_json.name,
            "checkpoint_npz": checkpoint_npz.name,
            "status_jsonl": status_path.name,
            "series_jsonl": (results / "series.jsonl").name,
            # v2.0 frame recorder: hash-bound manifest of frames/frame-NNNNNN.npz (None when recording was off)
            "frames": (frames_manifest(results) | {"config": getattr(frame_recorder_config(protocol), "to_dict", lambda: None)()})
            if list_frames(results) else None,
        },
        "provenance": sim.to_provenance() | {"runtime": artifacts.runtime_identity(), "config_sha256": artifacts.config_identity(config)},
        "simplifications": protocol["simplifications"],
        "claim_boundary": protocol.get("claim_boundary", (
            "development/screening PIC-MCC steady-state run; not preregistered; not validated against experiment; "
            "not a thruster performance prediction"
        )),
    }
    try:
        artifacts.write_canonical_json(results / "summary.json", summary)
    except Exception as error:
        # Fail closed but leave an honest terminal record: the stepping artifacts (maps, series, checkpoint-final)
        # are on disk, the run is NOT finished, and the reason is recorded so `finalize --recover-runner-stop`
        # can rebuild the summary from them without stepping or fabricating a stop.
        run_state.update({
            "finished": False,
            "finalization_error": {
                "utc": datetime.now(timezone.utc).isoformat(), "stop_reason_at_failure": stop_reason,
                "error": f"{type(error).__name__}: {error}", "artifacts_written": ["maps.npz", "series.npz", checkpoint_json.name, checkpoint_npz.name],
            },
        })
        artifacts.write_canonical_json(results / "run_state.json", run_state)
        raise
    artifacts.write_canonical_json(results / "run_state.json", run_state)
    _append_jsonl(status_path, {"event": "stop", "step": int(state.step), "time_s": float(state.time_s), "stop_reason": stop_reason})
    return results / "summary.json"


# -- the run ------------------------------------------------------------------

def run_steady_state(
    protocol: dict[str, Any],
    results: Path,
    *,
    backend: str = "warp-cuda",
    field_map: MagneticFieldMap | None = None,
    cross_sections: XenonCrossSections | None = None,
    max_steps: int | None = None,
    wall_budget_seconds: float | None = None,
    require_same_code: bool = True,
    protocol_path: Path = PROTOCOL_PATH,
    log: Callable[[str], None] = lambda text: print(text, flush=True),
    gpu_sample_interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    gpu_sampler_factory: Callable[[float], GpuUtilisationSampler] = lambda interval: GpuUtilisationSampler(interval_s=interval),
) -> Path:
    """Start or resume the run; returns the path of ``summary.json`` when the run stops.

    ``gpu_sample_interval_seconds`` is the cadence of the background ``nvidia-smi`` sampler (v2.0.2; default
    5 min, was a synchronous call per logged minute); the stepping thread never waits on it.
    """

    config = build_config(protocol, backend=backend)
    numerics = protocol["numerics"]
    rule = protocol["stopping_rule"]
    budget = protocol_budget(protocol)
    transit_time = float(budget["ion_transit_time_s"])
    checkpoint_every = int(numerics["checkpoint_every_steps"])
    window_steps = int(numerics["averaging_window_steps"])
    wall_budget = float(rule["wall_budget_seconds"]) if wall_budget_seconds is None else float(wall_budget_seconds)
    results.mkdir(parents=True, exist_ok=True)
    (results / "run.pid").write_text(f"{os.getpid()}\n", encoding="utf-8", newline="\n")

    t0 = time.perf_counter()
    field_map, cross_sections = load_inputs(config, field_map, cross_sections, protocol=protocol)
    xs_sha = cross_sections.payload_sha256 if cross_sections is not None else None
    setup_seconds = time.perf_counter() - t0

    sim = Simulation(config, field_map, cross_sections=cross_sections, backend=backend, step_graph=step_graph_flag(protocol))  # a-priori gate inside
    log(f"[steady-state] stability gate: {json.dumps(sim.stability.to_dict())}")
    log(f"[steady-state] mesh: {sim.masks.to_dict()}")
    if sim.neutrals is not None:
        log(f"[steady-state] neutral inventory: {json.dumps(sim.neutrals.to_dict())}")
    log(f"[steady-state] v1.4 options: {json.dumps(sim.to_provenance().get('v1_4_options'))}")
    if config.grid.geometry.has_plume:
        log(f"[steady-state] v2.0 options: {json.dumps(sim.to_provenance().get('v2_0_options'))}")
    plasma_volume = float(sim.masks.to_dict()["plasma_volume_m3"])
    # v2.0: the cathode region must sit on channel-connected field lines (fail-closed before any stepping)
    connectivity = cathode_connectivity_check(protocol, field_map, sim.masks)
    if connectivity is not None:
        log(f"[steady-state] cathode connectivity: {connectivity['connected_fraction']:.2f} of {connectivity['samples']} samples enter the channel "
            f"(required {connectivity['required_fraction']}); terminations {connectivity['terminations']}")
    # v2.0 frame recorder (default OFF): interval-averaged maps every cadence_steps, aligned with checkpoints/windows
    frame_config = frame_recorder_config(protocol)
    recorder: FrameRecorder | None = None
    if frame_config is not None:
        frame_config.validate_alignment(sync_steps=int(numerics["device_sync_steps"]), checkpoint_every_steps=checkpoint_every,
                                        window_steps=window_steps)
        recorder = FrameRecorder(results, frame_config, sim)
        log(f"[steady-state] frame recorder: every {frame_config.cadence_steps} steps ({frame_config.cadence_steps*config.dt_s*1e9:.1f} ns), "
            f"{frame_config.precision}, ~{estimate_frame_bytes(config.grid.node_shape, frame_config.precision)/1e6:.1f} MB/frame uncompressed")

    series_path = results / "series.jsonl"
    status_path = results / "status.jsonl"
    state_path = results / "run_state.json"
    run_state: dict[str, Any] = {"wall_seconds_total": 0.0, "sessions": [], "checkpoint_step": 0, "finished": False}
    records: list[dict[str, Any]] = []
    checkpoint = find_checkpoint(results)
    # the budget in force for this session is recorded because the CLI may raise it on a resume (attempt 8: 14400 -> 50400 s)
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "resumed_from_step": 0, "pid": os.getpid(),
               "wall_budget_seconds": wall_budget}
    if checkpoint is not None:
        # the field binding is verified through its platform-independent source identity; a resume on another CPU /
        # BLAS / OS is admitted only when the re-sampled map lies within the declared tolerance of the recorded anchor
        # (mode "numerical"), and the mode is recorded with the session (a numerical resume is not a bitwise replay)
        identity: dict[str, Any] = {}
        state = artifacts.load_checkpoint(checkpoint, config, field_sha256=field_map.sha256, field=field_map, cross_section_sha256=xs_sha,
                                          require_same_code=require_same_code, identity_report=identity)
        session["field_identity"] = identity["field"]
        sim.load_state(state)
        if state_path.is_file():
            run_state = json.loads(state_path.read_text(encoding="utf-8"))
        records = [r for r in _read_jsonl(series_path) if r["step"] <= state.step]
        _write_jsonl(series_path, records)  # drop records past the checkpoint (process died between sync and checkpoint)
        session["resumed_from_step"] = int(state.step)
        run_state["sessions"].append(session)
        # v2.1 hygiene (attempt 8 lesson): a resumed run is LIVE - the previous session's terminal block
        # (finished / stop_reason / finalized_from_step / finalization_recovery / finalization_error) is
        # demoted to ``history`` and the state is written BEFORE the first step (and before the resume is
        # logged), so a watcher never reads "finished: true, stop_reason: wall_clock_budget_reached" next
        # to an advancing checkpoint_step
        demoted = _demote_terminal_state(run_state, event="resume", step=int(state.step), utc=session["started_utc"],
                                         summary_present=(results / "summary.json").is_file())
        artifacts.write_canonical_json(state_path, run_state)
        _append_jsonl(status_path, {"event": "resume", "step": int(state.step), "time_s": float(state.time_s), "utc": session["started_utc"]})
        log(f"[steady-state] resumed from step {state.step} (t = {state.time_s*1e9:.1f} ns), {len(records)} series records kept"
            + ("" if demoted is None else f"; previous terminal state ({demoted.get('stop_reason')}) moved to run_state.history")
            + f"; field replay {identity['field']['mode']}"
            + ("" if identity["field"]["mode"] == "bitwise" else
               f" (max |dB| {identity['field']['comparison']['max_abs_diff_t']:.2e} T vs the recorded anchor; not a bitwise continuation)"))
        if recorder is not None:
            removed = recorder.reconcile(int(state.step))
            log(f"[steady-state] frames: {recorder.index} kept, {removed} past the checkpoint removed")
    else:
        run_state["sessions"].append(session)
    wall_before = float(run_state["wall_seconds_total"])

    stop_reason = "target_steps_reached"
    gate_error: str | None = None
    t_session = time.perf_counter()
    last_status_wall = t_session
    last_status_step = sim.backend.step_index
    last_print = t_session
    last_plateau: dict[str, Any] | None = None
    last_triad: dict[str, Any] | None = None
    # v2.0.2: nvidia-smi runs on a daemon thread at its own cadence; the loop only reads the shared last value
    gpu_sampler = gpu_sampler_factory(float(gpu_sample_interval_seconds)).start()
    log(f"[steady-state] gpu sampler: background, every {gpu_sampler.interval_s:.0f} s, timeout {gpu_sampler.timeout_s:.0f} s")

    def wall_total() -> float:
        return wall_before + (time.perf_counter() - t_session)

    def progress(record: SeriesRecord) -> None:
        nonlocal last_status_wall, last_status_step, last_print
        payload = record.to_dict()
        records.append(payload)
        _append_jsonl(series_path, payload)
        now = time.perf_counter()
        ms = 1e3 * (now - last_status_wall) / max(record.step - last_status_step, 1)
        last_status_wall, last_status_step = now, record.step
        _append_jsonl(status_path, status_from_record(payload, config, plasma_volume, wall_seconds_total=wall_total(),
                                                      ms_per_step=ms, plateau=last_plateau, triad=last_triad))
        if now - last_print > 60.0:
            last_print = now
            gpu_latest = gpu_sampler.latest()   # non-blocking: the last completed background sample (None if none / failed)
            extra = "" if gpu_latest is None else f" gpu={gpu_latest:.0f}%"
            extra += "" if record.neutral is None else f" n_g={record.neutral['density_per_m3']:.3g}"
            if record.neutral is not None and "net_utilisation" in record.neutral:
                extra += f" util={record.neutral['gross_utilisation']:.2f}/{record.neutral['net_utilisation']:.2f}"
            if record.peak_node is not None:
                extra += f" peak={record.peak_node['n_e_peak_per_m3']:.2e} cells/lD={record.peak_node['cells_per_debye']:.2f}"
                if record.peak_node.get("window") is not None:   # v2.0.3: the gated window statistic
                    window = record.peak_node["window"]
                    extra += f" win={window['cells_per_debye']:.2f}(w{window['window_steps']}{'' if window['gate_enforced'] else ' n/e'})"
            if last_triad is not None and last_triad.get("windowed_energy_residual_over_electrode_work") is not None:
                extra += f" res_w={last_triad['windowed_energy_residual_over_electrode_work']*100:+.1f}%"
            if last_triad is not None and last_triad.get("drift_members_arming") is not None:   # v2.1.1: drift-member latch state
                arming = last_triad["drift_members_arming"]
                extra += f" arm={'ARMED' if arming['armed'] else ('latched' if arming['latched'] else 'unlatched')}"
            if record.momentum is not None:  # v2.0
                extra += (f" T={record.momentum['thrust_total_n']*1e6:.1f} uN closure={record.momentum['closure_fraction']*100:.0f}%")
            if record.plume is not None:
                extra += f" phi_exit={record.plume['exit_plane_axis_potential_v']:.1f} V q_far={record.plume['charge_fraction_of_peak']:.3f}"
                if "far_field_window_steps" in record.plume:  # v2.0.2: window length / resolved nodes, unrestricted window and deposit statistics
                    extra += (f"(w{record.plume['far_field_window_steps']}/{record.plume['far_field_resolved_nodes']}n"
                              f" raw {record.plume['charge_fraction_of_peak_window_raw']:.3f} dep {record.plume['charge_fraction_of_peak_raw']:.3f})")
                elif "charge_fraction_of_peak_raw" in record.plume:  # v2.0.1 records
                    extra += f"(raw {record.plume['charge_fraction_of_peak_raw']:.3f}/{record.plume['far_field_resolved_nodes']}n)"
            log(f"[steady-state] step {record.step} t={record.time_s*1e6:.3f} us e={record.electrons} i={record.ions} "
                f"I_d={record.currents_a['discharge_a']*1e3:.2f} mA I_beam={record.currents_a['exit_ion_beam_a']*1e3:.2f} mA "
                f"S={record.currents_a['ionization_rate_per_s']:.3g}/s{extra} w_pe*dt={record.peak_omega_pe_dt:.3f} "
                f"{ms:.2f} ms/step wall={wall_total()/3600:.2f} h")

    step = sim.backend.step_index
    window_start = step
    completed_window: dict[str, np.ndarray] | None = None
    completed_range: tuple[int, int] | None = None
    while True:
        chunk = min(checkpoint_every, window_start + window_steps - step)
        if recorder is not None:   # stop at every frame boundary (the cadence divides checkpoint and window)
            chunk = min(chunk, recorder.steps_to_next_boundary(step))
        if max_steps is not None:
            chunk = min(chunk, max_steps - step)
        if chunk <= 0:
            stop_reason = "target_steps_reached"
            break
        try:
            sim.run(chunk, accumulate_from_step=window_start, progress=progress)
        except PIC2DStabilityError as error:
            gate_error = str(error)
            stop_reason = "runtime_stability_gate_stopped_run"
            log(f"[steady-state] fail-closed stop at step {sim.backend.step_index}: {gate_error}")
            break
        step = sim.backend.step_index
        if recorder is not None and recorder.due(step):
            recorder.capture(records[-1] if records else None)
        if step - window_start >= window_steps:
            completed_window = sim.diagnostic_arrays()
            completed_range = (window_start, step)
            sim.backend.reset_diagnostics()
            window_start = step
            if recorder is not None:
                recorder.on_window_reset()
        if recorder is not None and step % checkpoint_every != 0 and (max_steps is None or step < max_steps):
            continue   # frame boundary inside a checkpoint interval: no checkpoint / plateau evaluation yet
        # checkpoint after every checkpoint_every steps (every chunk without a recorder)
        save_checkpoint_atomic(results, sim, config, field_map, xs_sha)
        run_state.update({"wall_seconds_total": wall_total(), "checkpoint_step": step, "checkpoint_time_s": sim.state.time_s})
        if recorder is not None:
            run_state["frames_written"] = recorder.index
        if connectivity is not None:
            run_state["cathode_connectivity"] = connectivity
        artifacts.write_canonical_json(state_path, run_state)
        arrays = records_to_arrays(records)
        # v2.0 ignition gate: stop early (fail-closed) when S / N_e do not grow from the reference window
        last_ignition = evaluate_ignition(arrays, rule)
        if last_ignition is not None:
            run_state["ignition"] = last_ignition
            if last_ignition["failed"]:
                gate_error = last_ignition["reason"]
                stop_reason = "no_ignition"
                log(f"[steady-state] fail-closed stop at step {step}: {gate_error}")
                break
        last_plateau = evaluate_plateau(arrays["time_s"], arrays["current_discharge_a"], arrays["electrons"], rule, transit_time,
                                        arrays.get("neutral_density_per_m3"))
        last_triad = evaluate_triad(arrays, rule, transit_time)
        if last_triad is not None:
            # v1.4: the grid-heating triad is a fail-closed gate once enforced, and a plateau precondition always
            if last_triad["hard_failures"]:
                gate_error = "grid-heating triad gate: " + "; ".join(last_triad["hard_failures"])
                stop_reason = "grid_heating_triad_gate_stopped_run"
                log(f"[steady-state] fail-closed stop at step {step}: {gate_error}")
                break
            last_plateau["triad_soft_ok"] = last_triad["soft_ok"]
            last_plateau["reached"] = bool(last_plateau["reached"] and last_triad["soft_ok"])
        last_debye = evaluate_peak_debye_window(arrays, config)
        if last_debye is not None:
            # v2.0.3: the window-mode soft margin (2.5 cells per lambda_D) is a plateau precondition, never a stop
            last_plateau["peak_debye_soft_ok"] = last_debye["soft_ok"]
            last_plateau["peak_debye_cells_per_debye_window"] = last_debye["cells_per_debye_window_last"]
            last_plateau["reached"] = bool(last_plateau["reached"] and last_debye["soft_ok"])
        if last_plateau["reached"]:
            stop_reason = "plateau_reached_after_min_transit_times"
            break
        if wall_total() > wall_budget:
            stop_reason = "wall_clock_budget_reached"
            break
        if max_steps is not None and step >= max_steps:
            stop_reason = "target_steps_reached"
            break

    wall_session = time.perf_counter() - t_session
    run_state.update({"wall_seconds_total": wall_before + wall_session, "finished": True, "stop_reason": stop_reason})
    gpu_sampler.stop(join_timeout_s=1.0)    # never waits for a hung nvidia-smi: the thread is a daemon
    gpu_samples = gpu_sampler.snapshot()
    partial = sim.diagnostic_arrays()
    if int(partial["window_steps"][0]) >= window_steps // 2 or completed_window is None:
        maps, window_range = partial, (window_start, sim.backend.step_index)
    else:
        maps, window_range = completed_window, completed_range  # type: ignore[assignment]
    summary_path = write_final_artifacts(
        protocol=protocol, protocol_path=protocol_path, results=results, sim=sim, config=config, field_map=field_map,
        xs_sha=xs_sha, records=records, maps=maps, window_range=window_range, maps_kind="window_average",
        stop_reason=stop_reason, gate_error=gate_error, run_state=run_state, session=session,
        setup_seconds=setup_seconds, wall_session=wall_session, gpu_samples=gpu_samples, gpu_sampler=gpu_sampler.summary(),
    )
    state = sim.state
    log(f"[steady-state] done: {state.step} steps, t = {state.time_s*1e6:.3f} us, {stop_reason}; summary at {summary_path}")
    return summary_path


# -- finalize (no stepping) -----------------------------------------------------

def finalize(
    protocol: dict[str, Any],
    results: Path,
    *,
    backend: str = "warp-cuda",
    field_map: MagneticFieldMap | None = None,
    cross_sections: XenonCrossSections | None = None,
    stop_reason: str = "finalized_from_checkpoint",
    protocol_path: Path = PROTOCOL_PATH,
    log: Callable[[str], None] = lambda text: print(text, flush=True),
    allow_refinalize: bool = False,
    recover_runner_stop: bool = False,
) -> Path:
    """Write summary/maps/series from the latest checkpoint and the series history without stepping.

    The device-side window accumulators die with the process, so the maps are the
    instantaneous single-sample maps of the checkpoint (``maps_kind =
    "instantaneous_checkpoint"``; flux and ionisation maps are zero).  The checkpoint
    is loaded with the code-identity check relaxed (no dynamics are computed);
    ``backend`` must be the one the run used (the Poisson method is part of the
    config identity).

    A run that the runner itself ended (plateau, wall budget, stability gate) already
    has its window-average artifacts; finalizing it again would *downgrade* the maps
    to instantaneous ones and rewrite the stop reason, so that is refused unless
    ``allow_refinalize`` is set.

    ``recover_runner_stop`` (fail-closed) rebuilds ``summary.json`` for a run whose runner
    DID stop and wrote ``maps.npz`` / ``series.npz`` / ``checkpoint-final.*`` but crashed
    before the summary (plume attempt 7: a NaN GPU-utilisation sample made the summary
    non-canonical).  It reuses the window-average maps verbatim (``maps_kind`` stays
    ``window_average``; the rebuilt bytes must hash to the sidecar the runner wrote) and
    the final checkpoint, and it accepts only a ``stop_reason`` whose evidence is on disk:
    ``wall_clock_budget_reached`` needs ``run_state.wall_seconds_total`` above the
    protocol's budget, ``plateau_reached_after_min_transit_times`` needs the plateau rule
    to hold on the recorded series.  Nothing is stepped and no stop is invented.
    """

    checkpoint = find_checkpoint(results)
    if checkpoint is None:
        raise PIC2DValidationError(f"no checkpoint to finalize under {results}")
    state_path = results / "run_state.json"
    if state_path.is_file() and (results / "summary.json").is_file() and not allow_refinalize:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("finished") and "finalized_from_step" not in previous:
            raise PIC2DValidationError(
                f"{results.name} was already finished by the runner ({previous.get('stop_reason')}) with window-average "
                "maps; finalize would replace them with instantaneous checkpoint maps (use --allow-refinalize to override)"
            )
    recovery: dict[str, Any] | None = None
    if recover_runner_stop:
        recovery = _runner_stop_recovery_preflight(results, protocol, stop_reason)
        checkpoint = results / "checkpoint-final.json"
    config = build_config(protocol, backend=backend)
    t0 = time.perf_counter()
    field_map, cross_sections = load_inputs(config, field_map, cross_sections, protocol=protocol)
    xs_sha = cross_sections.payload_sha256 if cross_sections is not None else None
    sim = Simulation(config, field_map, cross_sections=cross_sections, backend=backend, step_graph=step_graph_flag(protocol))
    identity: dict[str, Any] = {}
    state = artifacts.load_checkpoint(checkpoint, config, field_sha256=field_map.sha256, field=field_map, cross_section_sha256=xs_sha,
                                      require_same_code=False, identity_report=identity)
    sim.load_state(state)
    setup_seconds = time.perf_counter() - t0
    records = [r for r in _read_jsonl(results / "series.jsonl") if r["step"] <= state.step]
    _write_jsonl(results / "series.jsonl", records)
    frame_config = frame_recorder_config(protocol)
    if frame_config is not None and list_frames(results):
        removed = FrameRecorder(results, frame_config, sim).reconcile(int(state.step))
        log(f"[steady-state] frames past the checkpoint removed: {removed}")
    state_path = results / "run_state.json"
    run_state: dict[str, Any] = {"wall_seconds_total": 0.0, "sessions": [], "checkpoint_step": int(state.step), "finished": False}
    if state_path.is_file():
        run_state = json.loads(state_path.read_text(encoding="utf-8"))
    session = {"started_utc": datetime.now(timezone.utc).isoformat(), "resumed_from_step": int(state.step), "pid": os.getpid(), "finalize_only": True,
               "field_identity": identity["field"]}
    # v2.1 hygiene: whatever terminal block the file carried (an earlier finalization, a recovery, a recorded
    # finalization_error, or the stale block of a run that was resumed by an older runner) goes to history;
    # the terminal state written below belongs to THIS finalization only
    _demote_terminal_state(run_state, event="finalize", step=int(state.step), utc=session["started_utc"],
                           summary_present=(results / "summary.json").is_file())
    if recovery is not None:
        if int(state.step) != int(recovery["step"]):
            raise PIC2DValidationError("checkpoint-final step differs from its metadata")   # pragma: no cover - load_checkpoint binds it
        session["recovered_runner_stop"] = True
        run_state["sessions"].append(session)
        run_state.update({
            "finished": True, "stop_reason": stop_reason, "finalized_from_step": int(state.step),
            "finalization_recovery": {
                "mode": "runner_stop_artifacts_reused", "recovered_utc": session["started_utc"], "stop_reason_evidence": recovery["evidence"],
                "reused": ["maps.npz (window average, byte-identical to the runner's sidecar)", "checkpoint-final.json/.npz"],
                "original_error": recovery["original_error"],
                "wall_seconds_note": "wall_seconds_total is the value recorded at the last checkpoint before the stop; the seconds spent "
                                     "writing the stop artifacts are not included",
            },
        })
        maps = recovery["maps"]
        window_range = recovery["window_range"]
        maps_kind = "window_average"
        log(f"[steady-state] recovering the runner stop of {results.name} at step {state.step} (t = {state.time_s*1e6:.3f} us): "
            f"{stop_reason} ({recovery['evidence']}); window {window_range}, {len(records)} records")
    else:
        run_state["sessions"].append(session)
        run_state.update({"finished": True, "stop_reason": stop_reason, "finalized_from_step": int(state.step)})
        maps = instantaneous_maps(config, sim.masks, state)
        window_range = (int(state.step), int(state.step))
        maps_kind = "instantaneous_checkpoint"
        log(f"[steady-state] finalizing {results.name} from step {state.step} (t = {state.time_s*1e6:.3f} us), {len(records)} records")
    summary_path = write_final_artifacts(
        protocol=protocol, protocol_path=protocol_path, results=results, sim=sim, config=config, field_map=field_map,
        xs_sha=xs_sha, records=records, maps=maps, window_range=window_range,
        maps_kind=maps_kind, stop_reason=stop_reason, gate_error=None, run_state=run_state,
        session=session, setup_seconds=setup_seconds, wall_session=0.0, gpu_samples=[],
    )
    if recovery is not None:
        for name, expected in (("maps.npz", recovery["maps_sha256"]), ("checkpoint-final.npz", recovery["checkpoint_npz_sha256"])):
            rewritten = json.loads((results / f"{name}.sha256.json").read_text(encoding="utf-8"))["byte_sha256"]
            if rewritten != expected:
                raise PIC2DValidationError(f"recovered {name} does not hash to the runner's sidecar ({rewritten[:12]} != {expected[:12]})")
    return summary_path


def _runner_stop_recovery_preflight(results: Path, protocol: dict[str, Any], stop_reason: str) -> dict[str, Any]:
    """Fail-closed evidence check for ``finalize(recover_runner_stop=True)``.

    Returns the verified window-average maps, the window range the runner used, the final
    checkpoint step, the stop-reason evidence and the recorded finalization error.
    """

    if (results / "summary.json").is_file():
        raise PIC2DValidationError(f"{results.name} already has a summary.json; nothing to recover")
    state_path = results / "run_state.json"
    if not state_path.is_file():
        raise PIC2DValidationError(f"{results.name} has no run_state.json; the runner never checkpointed")
    run_state = json.loads(state_path.read_text(encoding="utf-8"))
    if run_state.get("finished"):
        raise PIC2DValidationError(f"{results.name} is marked finished ({run_state.get('stop_reason')}); nothing to recover")
    final_json = results / "checkpoint-final.json"
    if not final_json.is_file() or not (results / "maps.npz").is_file() or not (results / "maps.npz.sha256.json").is_file():
        raise PIC2DValidationError(f"{results.name}: the runner did not reach its stop (checkpoint-final.json / maps.npz missing)")
    final_meta = artifacts.read_canonical_json(final_json)
    latest = find_checkpoint(results)
    if latest is not None and int(artifacts.read_canonical_json(latest)["step"]) != int(final_meta["step"]):
        raise PIC2DValidationError("checkpoint-final and checkpoint-latest disagree on the step; the run did not stop at this checkpoint")
    step = int(final_meta["step"])
    if int(run_state.get("checkpoint_step", -1)) != step:
        raise PIC2DValidationError(f"run_state checkpoint_step {run_state.get('checkpoint_step')} differs from checkpoint-final step {step}")
    maps_sha = json.loads((results / "maps.npz.sha256.json").read_text(encoding="utf-8"))["byte_sha256"]
    maps = artifacts.read_npz(results / "maps.npz", expected_sha256=maps_sha)
    # write_final_artifacts re-serialises the maps through artifacts.write_npz; prove the round trip is byte-exact BEFORE
    # touching the file (deterministic uncompressed savez over sorted keys)
    buffer = io.BytesIO()
    np.savez(buffer, **{key: np.ascontiguousarray(maps[key]) for key in sorted(maps)})
    if hashlib.sha256(buffer.getvalue()).hexdigest() != maps_sha:
        raise PIC2DValidationError("maps.npz does not round-trip byte-exactly through write_npz; refusing to rewrite it")
    # the runner's window choice: the current window if at least half full, else the last completed one
    window_steps = int(protocol["numerics"]["averaging_window_steps"])
    used = int(maps["window_steps"][0])
    if used < window_steps:
        window_range = (step - used, step)
    else:
        end = step - step % window_steps
        window_range = (end - window_steps, end)
    if window_range[0] < 0 or window_range[1] > step or window_range[1] - window_range[0] != used:
        raise PIC2DValidationError(f"cannot reconstruct the averaging window ({used} steps) at step {step}")
    records = [r for r in _read_jsonl(results / "series.jsonl") if r["step"] <= step]
    rule = protocol["stopping_rule"]
    if stop_reason == "wall_clock_budget_reached":
        wall = float(run_state.get("wall_seconds_total", 0.0))
        budget = float(rule["wall_budget_seconds"])
        if not wall > budget:
            raise PIC2DValidationError(f"recorded wall time {wall:.0f} s does not exceed the budget {budget:.0f} s; refusing '{stop_reason}'")
        evidence = f"run_state.wall_seconds_total {wall:.1f} s > wall_budget_seconds {budget:.0f} s at checkpoint step {step}"
    elif stop_reason == "plateau_reached_after_min_transit_times":
        if not records:
            raise PIC2DValidationError("no series records; cannot verify the plateau stop")
        arrays = records_to_arrays(records)
        transit = float(protocol_budget(protocol)["ion_transit_time_s"])
        plateau = evaluate_plateau(arrays["time_s"], arrays["current_discharge_a"], arrays["electrons"], rule, transit,
                                   arrays.get("neutral_density_per_m3"))
        triad = evaluate_triad(arrays, rule, transit)
        debye_window = evaluate_peak_debye_window(arrays, build_config(protocol, backend="cpu"))
        if not plateau["reached"] or (triad is not None and not triad["soft_ok"]) or (debye_window is not None and not debye_window["soft_ok"]):
            raise PIC2DValidationError(f"the recorded series does not satisfy the plateau rule; refusing '{stop_reason}'")
        evidence = f"plateau rule holds on the recorded series at step {step} ({plateau['transit_times_elapsed']:.2f} transits)"
    else:
        raise PIC2DValidationError(f"stop reason '{stop_reason}' has no on-disk evidence; recovery accepts wall_clock_budget_reached or "
                                   "plateau_reached_after_min_transit_times")
    original_error = run_state.get("finalization_error")
    if original_error is None:
        err_path = results / "run.err"
        tail = err_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:] if err_path.is_file() else []
        original_error = {"error": tail[0] if tail else None, "source": "run.err (last line)" if tail else "not recorded"}
    return {"maps": maps, "maps_sha256": maps_sha, "checkpoint_npz_sha256": str(final_meta["arrays_sha256"]), "window_range": window_range,
            "step": step, "evidence": evidence, "original_error": original_error}


# -- status -----------------------------------------------------------------

def status(results: Path = RESULTS, protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = protocol or load_protocol()
    lines = _read_jsonl(results / "status.jsonl")
    samples = [line for line in lines if "event" not in line]
    if not samples:
        return {"status": "no samples yet"}
    last = samples[-1]
    transit = float(protocol_budget(protocol)["ion_transit_time_s"])
    steps_per_transit = transit / float(protocol["numerics"]["dt_s"])
    recent = samples[-50:]
    ms = float(np.nanmean([s["ms_per_step"] for s in recent if s["ms_per_step"] is not None]))
    remaining = {f"{k}_transit_times": {"steps": int(k * steps_per_transit), "hours_from_now": (k * steps_per_transit - last["step"]) * ms / 3.6e6}
                 for k in (3, 5, 10)}
    pid_file = results / "run.pid"
    state_path = results / "run_state.json"
    run_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None
    return {
        "last": last,
        "samples": len(samples),
        "recent_ms_per_step": ms,
        "transit_times_elapsed": last["time_s"] / transit,
        "projection": remaining,
        "pid": int(pid_file.read_text().strip()) if pid_file.is_file() else None,
        # v2.1: the live flag comes from run_state.json (a resumed run reuses the directory, so a summary.json of the
        # superseded stop may sit next to an advancing checkpoint); falls back to the summary for pre-run_state runs
        "finished": bool(run_state["finished"]) if run_state is not None else (results / "summary.json").is_file(),
        "stop_reason": None if run_state is None else run_state.get("stop_reason"),
        "history_entries": 0 if run_state is None else len(run_state.get("history", [])),
        "frames_written": len(list_frames(results)),
    }


def main(argv: list[str] | None = None, *, protocol_path: Path = PROTOCOL_PATH, results: Path = RESULTS) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=None, help="named variant from protocol['variants'] (results in results-<case>)")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--backend", default="warp-cuda")
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--wall-budget-seconds", type=float, default=None)
    run_parser.add_argument("--ignore-code-identity", action="store_true", help="resume even if the package code hash changed")
    run_parser.add_argument("--gpu-sample-interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS,
                            help="cadence of the background nvidia-smi utilisation sampler (v2.0.2; default 300 s; the step loop never waits on it)")
    fin = sub.add_parser("finalize")
    fin.add_argument("--backend", default="warp-cuda", help="the backend the run used (part of the config identity)")
    fin.add_argument("--stop-reason", default="finalized_from_checkpoint")
    fin.add_argument("--allow-refinalize", action="store_true", help="re-finalize a run the runner already finished (downgrades the maps)")
    fin.add_argument("--recover-runner-stop", action="store_true",
                     help="rebuild summary.json for a runner stop whose final write crashed (reuses maps.npz + checkpoint-final; "
                          "--stop-reason must be evidenced on disk: wall_clock_budget_reached or plateau_reached_after_min_transit_times)")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    protocol, results_name = apply_case(load_protocol(protocol_path), args.case, load_variants(protocol_path))
    results = results.parent / results_name
    if args.command == "run":
        run_steady_state(protocol, results, backend=args.backend, max_steps=args.max_steps, wall_budget_seconds=args.wall_budget_seconds,
                         require_same_code=not args.ignore_code_identity, protocol_path=protocol_path,
                         gpu_sample_interval_seconds=args.gpu_sample_interval_seconds)
    elif args.command == "finalize":
        finalize(protocol, results, backend=args.backend, stop_reason=args.stop_reason, protocol_path=protocol_path,
                 allow_refinalize=args.allow_refinalize, recover_runner_stop=args.recover_runner_stop)
    else:
        print(json.dumps(status(results, protocol), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
