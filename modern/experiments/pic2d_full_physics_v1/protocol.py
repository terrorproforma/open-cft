"""Composition of the full-physics run protocols on the steady-state v4 (33 um) template (roadmap R4 + R5 + the R1-R5 combination).

Each case is the ss-v4 protocol (``pic2d_cft_steady_state_v4/protocol.json``: reference design, 90 x 720 cells at 33.33 um, dt 1.4 ps,
W 26 666.7, seed 20260903, frames ON) with the v2.0.6 gates (accumulated-particle-step Debye floor 64 000, W-corrected ledger), K = 5, the
model v2.1.1 drift-member ARMING LATCH and the IGNITION GATE of the alpha-series amendment 1 (both stopping-rule keys outside
``config_sha256``), and exactly these physics changes:

* ``coulomb``: ``numerics.coulomb`` = model v2.4.0 ``coulomb_v1`` (e-e + e-i Takizuka-Abe / Nanbu every 10 steps, i-i off); everything else
  legacy (alpha = 0, 0-D inventory, legacy collision set, absorbing wall) - R4 isolated against the recorded ss-v4 plateau;
* ``neutrals-spatial``: ``operating_point.neutrals`` = model v2.5.0 ``neutrals_spatial_v1`` + ``metastables_v1`` at time acceleration F = 1
  (physical neutral time) REPLACING the 0-D inventory, with the MCC null-collision ceiling raised above the Knudsen anode density (fail-closed);
  the metastable pool's level-resolved branching REQUIRES the v2.3.0 ``xe_collision_set_v2`` (the code refuses ``metastables_v1`` on the lumped
  legacy set), so this case carries the collision set v2 as well - stated everywhere; SEE off, Coulomb off, alpha = 0;
* ``neutrals-spatial-F10``: the same at F = 10 - the TIME-ACCELERATION QUALIFICATION pair (predeclared: the plateau scalars must agree with
  F = 1 inside the particle band for F to be usable in later runs);
* ``full-physics-alpha0``: SEE(BN) + xe_collision_set_v2 + coulomb + neutrals_spatial (+ metastables, F = 1) at alpha = 0;
* ``full-physics-alpha1over16`` / ``full-physics-alpha0.345``: the same with the v2.1.0 Bohm perpendicular-rotation closure at alpha = 1/16 and
  0.345 - the sustain question: a Bohm-leaky discharge extinguished at the dilute 0-D gas (n_g 5.5e19; alpha-series launch 1, 0916a4f8) and
  sustained marginally at Brandt's 2e20 (ext-val bohm-0.4); does the Knudsen-profile gas (channel mean ~2.5e20) sustain it here?

The reference point of every shift table is the RECORDED ss-v4 plateau (0d228ad2), which fails its own acceptance (b) at +2.46 % on the
corrected ledger - stated in every sealed protocol.  Everything else (geometry, dt, grid, W, seed, cadences, plateau rule, the v2.0.3 gate
thresholds) is byte-for-byte the v4 protocol; ``tests/pic2d/test_pic2d_full_physics_v1.py`` pins that.
"""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.orbit_mc.artifacts import canonical_bytes
from cft_revival.pic2d.sensitivity import ANOMALOUS_MODEL_ROTATION, BOHM_ALPHA_SERIES
from experiments.pic2d_anomalous_transport_v1 import protocol as at_protocol
from experiments.pic2d_physics_effects_v1 import protocol as pe_protocol
from experiments.pic2d_physics_effects_v1.protocol import (
    ABSOLUTE_BAND as PE_ABSOLUTE_BAND,
)
from experiments.pic2d_physics_effects_v1.protocol import (
    COLLISION_SET_V2_BLOCK,
    CUSP_HALF_WIDTH_M,
    CUSP_PLANES_M,
    IEDF_FRACTION_BAND,
    PARTICLE_BAND,
    REFERENCE_CORRECTED_RESIDUAL,
    SEE_BN_BLOCK,
    STEPS_TO_3_TRANSITS,
    channel_wall_cells,
    iedf_low_energy_fraction,
    load_v4_protocol,
    wall_area_m2,
    wall_power_and_ion_energy,
)

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
V4_DIR = MODERN / "experiments" / "pic2d_cft_steady_state_v4"
V4_PROTOCOL_PATH = V4_DIR / "protocol.json"
V4_RESULTS = V4_DIR / "results"
PE_RESULTS = MODERN / "experiments" / "pic2d_physics_effects_v1" / "results"
AT_RESULTS = MODERN / "experiments" / "pic2d_anomalous_transport_v1" / "results"
PROTOCOLS_DIR = HERE / "protocols"
CAMPAIGN_PROTOCOL_PATH = HERE / "protocol.json"

SCHEMA_VERSION = "cft-revival.pic2d-full-physics-v1.protocol/0.1.0"
CASE_SCHEMA_VERSION = "cft-revival.pic2d-full-physics-v1.case-protocol/0.1.0"
EXPERIMENT_ID = "pic2d-full-physics-v1"
MODEL_VERSION = ("pic2d model v2.5.0 physics set on the ss-v4 template: v1.3 injection / seed / grid, per case: v2.1.0 anomalous transport "
                 "(bohm_perpendicular_rotation), v2.2.0 see_dielectric_v1 (BN), v2.3.0 xe_collision_set_v2, v2.4.0 coulomb_v1, v2.5.0 neutrals_spatial_v1 + "
                 "metastables_v1 (replacing the 0-D inventory); v2.0.6 gates (window-mode peak-Debye gate with the accumulated-particle-step floor, windowed "
                 "residual-power gate on the W-corrected ledger), v2.0.5 K = 5, model v2.1.1 drift-member arming latch + the v2.0 ignition gate")
MODEL_SPEC = ("modern/spec/pic2d/pic2d-model-v1.3.json (physics) + pic2d-model-v2.0.json (gates_v2_0, anomalous_transport_v2_1_0, triad_drift_arming_v2_1_1) + "
              "pic2d-model-v2.2.json (see_dielectric_v1) + pic2d-model-v2.3.json (xe_collision_set_v2) + pic2d-model-v2.4.json (coulomb_v1) + "
              "pic2d-model-v2.5.json (neutrals_spatial_v1, metastables_v1)")
REFERENCE_CASE = "ss-v4"
KNUDSEN_ANODE_DENSITY_PER_M3 = 5.45e20          # spec v2.5.0 / R5 shakedown: the closed-end free-molecular profile at the ss-v4 feed peaks here on the anode face
KNUDSEN_CHANNEL_MEAN_PER_M3 = 2.49e20           # R5 shakedown trailing mean (55092f4c): 4.5 x the 0-D fixed point 5.5e19
MCC_CEILING_SPATIAL_PER_M3 = 1.5e21             # 2.75 x the Knudsen anode density: see MCC_CEILING_NOTE
NEUTRAL_MACRO_WEIGHT = 2.2e7                    # ~4.0 M macro-neutrals at the Knudsen profile (~60 per 33 um cell; 825 x the ion macro weight)
NEUTRAL_SUBSTEP_STEPS = 200                     # one neutral sub-step per device-sync interval (0.28 ns of plasma time)
NEUTRAL_WALL_TEMPERATURE_K = 500.0              # the v1.4 recycling convention (mini-sweep / R5 shakedown); feed gas 300 K
CENTROID_BAND_M = 1.0e-3                        # DECLARED absolute detection band for the ionisation centroid (30 cells; no replicate exists)
F_QUALIFICATION_FACTOR = 10.0

# the two sealed physics blocks of the alpha-series amendment 1 (the numeric keys are identical; the notes are this campaign's)
DRIFT_MEMBERS_ARMING: dict[str, Any] = {
    **{k: at_protocol.DRIFT_MEMBERS_ARMING[k] for k in ("min_transit_times", "settle_quantity", "settle_drift_max", "settle_check_cadence_steps")},
    "note": ("model v2.1.1 (from the alpha-series amendment 1, 33be2a89): the triad's DRIFT members (trailing-20 % drifts of S, T_e,dense and the resolved "
             "omega_pe dt; hard 0.25) are enforced only once >= 2.0 ion transits have elapsed AND the trailing-20 % I_d drift has read < 0.05 at a 40 000-step "
             "checkpoint at or after 2.0 transits (a settled-once latch). Every case here changes the operating point or the closure, so the v1.4 rule "
             "(enforced_after_transit_times 1.0, calibrated on alpha = 0 plateaus at the 0-D gas) is superseded exactly as in the alpha-series: a discharge "
             "re-equilibrating to a new state must not be stopped for moving. The physics protections are unchanged and independent of the arming: the "
             "one-sided windowed residual-POWER gate (>= 5 % of the electrode work from the first complete 400 000-step window) and the window-mode "
             "peak-Debye hard gate (pi cells per lambda_D on the accumulated-floor peak). Calibration (e47ae78a): ss-v4's latch closes at 2.66 transits, "
             "047 / 009 / 056-L2 inside 0.05 from 2.0 transits; attempt 8 and the ext-val launch 1 are still stopped by the residual member"),
}
IGNITION_GATE: dict[str, Any] = {
    **{k: copy.deepcopy(at_protocol.IGNITION_GATE[k]) for k in ("reference_window_s", "check_window_s", "checks")},
    "note": ("the v2.0 runner ignition gate as sealed by the alpha-series amendment 1 (33be2a89): fail-closed stop_reason no_ignition, evaluated at every "
             "checkpoint from the series - trailing 0.15 us means of S and N_e over their 0.05-0.2 us reference means (the run's OWN post-seed reference, so "
             "the ratios transfer to the denser Knudsen gas and to the SEE wall) must read N_e >= 0.6 and S >= 0.3 of the reference at 1.0 us, >= 0.6 / "
             ">= 0.4 at 2.0 us. Calibration: accepted 33 um plateaus at the 0-D gas 1.31-1.44 / 0.96-1.17 at 1.0 us (margins >= 2.2x / 2.4x); the "
             "extinguished alpha-1over16 launch 1 read 0.45 / 0.37 at 1.0 us and would have stopped at 1.008 us. Here the gate is THE predeclared reading "
             "of the sustain hypothesis of the full-physics alpha cases: an EXTINCTION (stop_reason no_ignition) is a valid recorded outcome of the closure "
             "at this operating point - never a reason to adjust the seed, the injection or alpha. The 100 000-step shakedowns reach 0.14 us only: they "
             "record the direction of N_e and S over the reference window (the 1/16 extinction was already visible at 0.1 us: N_e -24 %), not the gate"),
}

# -- the physics blocks exactly as the runner reads them --------------------------------------------------------------------------------
COULOMB_BLOCK: dict[str, Any] = {
    "enabled": True, "electron_electron": True, "electron_ion": True, "ion_ion": False, "cycle_steps": 10, "coulomb_log_floor": 2.0, "min_temperature_ev": 0.01,
}
COULOMB_NOTE = ("model v2.4.0 coulomb_v1 (spec pic2d-model-v2.4.json; R4, audit gap d): every cycle_steps = 10 steps, after push / absorb and before the ion MCC "
                "/ anomalous / electron MCC, per cell: e-e Takizuka-Abe random-permutation pairing (odd cells -> triplet at dt_c / 2), e-i every electron once "
                "against ion (l + shift) mod N_i at the field ion density; Nanbu 1997 cumulative scattering angle (exact small-s mean; Perez-2012 fit to 3; "
                "3 e^-s to 6; isotropic beyond); exact centre-of-mass rotation (pair momentum and classical energy to round-off); NRL Coulomb logarithm from "
                "the cell's n and T with floors 0.01 eV / 2.0; dt_c = 14 ps keeps the peak-cell s ~ 4e-5 at 1e18 (~4e-4 at 1e19). i-i OFF (nu_ii / nu_ee ~ 1e-3; "
                "nu_ii tau_transit << 1 for the accelerated population). GPU: a per-cycle cell-sorted slot permutation (particles never moved; the index-keyed "
                "RNG contract holds: Coulomb-off is bitwise the v2.2.0 pin, graph = direct bitwise with it on); seed-table column 5 / CPU stream 6. Ledger: "
                "pz_coulomb (~0), ke_coulomb_j (O(v^2/c^2) remainder); series coulomb block with the operator's pair-mean deflection rates <s>/dt_c (a 1/g^3-"
                "weighted mean, ~13x the Spitzer rate, recorded as the realised deflection statistic) AND the NRL Spitzer electron collision rate "
                "2.91e-6 n lnL T^-3/2 at the peak node (the audit's gap-(d) definition, the comparable number) with its ratio to nu_en. Every field of this "
                "block enters config_sha256 through CoulombConfig.to_dict. Box shakedown (82255081, non-evidentiary, 100k steps on the R3 protocol): cycle "
                "4.49 ms (sort 1.14, pairs 3.34) = +0.48 ms/step amortised (+7.3 % contended); Spitzer nu_e at the peak cell 2.8e5 /s vs nu_en 1.17e7 -> 0.024 "
                "in the seed transient (~0.3 scaled to the plateau peak; the audit's 0.15-0.4 is to be read from THIS run); S -0.5 %, I_d -0.4 % (shot noise)")

MCC_CEILING_NOTE = (
    f"MCC null-collision ceiling n_g0 ONLY (v2.5.0 neutrals_spatial_v1; the instantaneous density is the published per-cell field of the neutral particles): "
    f"the initial Knudsen closed-end profile at the ss-v4 feed peaks at {KNUDSEN_ANODE_DENSITY_PER_M3:.3g} m^-3 on the anode face (10 x the exit value "
    f"Q_in / c = 5.5e19 for the 2 mm bore / 3 mm exit), so the ceiling is raised from the 0-D protocol's 5.5e19 to {MCC_CEILING_SPATIAL_PER_M3:.3g} = 2.75 x the "
    f"Knudsen anode density, FAIL-CLOSED: the published density is clamped there and the run ends (PIC2DStabilityError at the next series record) when more "
    f"than max_ceiling_violation_fraction 1e-3 of the plasma cell-substeps of an interval are clamped. Why 2.75 x and not the R5 shakedown's 1e21 (1.8 x): the "
    f"smallest-volume axis cells (pi dr^2 dz = 1.15e-13 m^3) hold ~2.9 macro-neutrals at W_n {NEUTRAL_MACRO_WEIGHT:.3g} and the Knudsen anode density, so their "
    f"nearest-cell deposit is Poisson with sigma / mu ~ 0.6; at 1e21 (>= 6 macro-neutrals) ~7 % of the ~300 dense axis cells clamp per deposit = the "
    f"3.5e-4 violation fraction the R5 shakedown recorded (a 3x margin under the 1e-3 limit over a 100k-step run; too thin for a 12-16 h run); at "
    f"{MCC_CEILING_SPATIAL_PER_M3:.3g} (>= 8) ~1 % clamp -> ~5e-5, a 20x margin, while a REAL excess of the gas above 2.75 x the Knudsen anode density (an "
    f"inventory pile-up) still ends the run. The ceiling sets only the null-collision candidate rate (nu_max dt ~ 3e-3 per step: ~0.3 % of the electrons are "
    f"candidates each step, the operator stays exact) and the metastable ceiling (ceiling_fraction 0.05 x n_g0 = 7.5e19 vs a local metastable density "
    f"<= 11 % of the local ground density in the R5 shakedown). neutral_density_per_m3 enters config_sha256 (MCCConfig), so both F cases share it")


def neutrals_block(time_acceleration: float) -> dict[str, Any]:
    """The ``operating_point.neutrals`` block (model v2.5.0) at the declared time acceleration F (every non-note field enters ``config_sha256``)."""

    v4 = load_v4_protocol()
    feed = float(v4["operating_point"]["neutral_inventory"]["feed_atoms_per_s"])
    f_acc = float(time_acceleration)
    return {
        "model": "neutrals_spatial_v1",
        "feed_atoms_per_s": feed,
        "feed_note": ("the ss-v4 feed Q_in = 8.551e16 atoms/s (0.01864 mg/s) UNCHANGED - the operating point of every recorded plateau; the spatial model at this feed "
                      f"carries the Knudsen closed-end profile (anode {KNUDSEN_ANODE_DENSITY_PER_M3:.3g} -> exit 7.0e19 m^-3, channel mean {KNUDSEN_CHANNEL_MEAN_PER_M3:.3g} = "
                      "4.5 x the 0-D fixed point 5.5e19): the 0-D inventory EQUATED THE WHOLE CHANNEL TO THE EXIT DENSITY (R5 finding, 55092f4c). The same feed at a "
                      "different gas distribution is a DIFFERENT OPERATING POINT: this campaign measures what that does to the plasma"),
        "macro_weight": NEUTRAL_MACRO_WEIGHT,
        "macro_weight_note": ("~8.7e13 atoms in the channel at the Knudsen profile -> ~4.0 M macro-neutrals (~60 per 33 um cell), ~0.5 GB of device memory (declared; "
                              "measured in preflight-<case>.json memory.device_used_by_loaded_run_bytes against the coulomb case); 825 x the ion macro weight 26 666.7"),
        "substep_steps": NEUTRAL_SUBSTEP_STEPS,
        "substep_note": "one neutral sub-step per device-sync / series interval (200 steps = 0.28 ns of plasma time; the sinks the MCCs book over the interval are applied at the sub-step)",
        "time_acceleration": f_acc,
        "time_acceleration_note": (
            f"DECLARED numerical parameter F = {f_acc:g}: neutral time = F x plasma time (the spatial analogue of the 0-D tau_g); the neutral ledger is kept in neutral time and "
            "the real-time plasma rates are ledger / F. F = 1 is PHYSICAL neutral time. WHY F EXISTS: the physical neutral relaxation of the channel gas (tube residence "
            "V / c ~ 0.22 ms; the free-molecular re-equilibration of the Knudsen profile to the depleted steady state 0.2-2 ms) is 30-300 x longer than the whole run "
            "(3 ion transits = 7.2 us), so at F = 1 the gas over the run is the initial Knudsen profile minus a depletion of ~0.3 % (S ~ 4e16 /s x 7 us against 8.6e13 "
            "atoms): a QUASI-FROZEN gas with the right profile, not the self-consistent neutral steady state (that needs F ~ 100-300). WHAT F DISTORTS: the neutral "
            "RESPONSE TIME - depletion, recycling refill, CEX fast-neutral flight, the metastable pool's filling (lifetime ~ the wall transit, 5-10 us of neutral time, so at "
            "F = 1 the pool is still filling at 7 us while at F = 10 it is quasi-steady) and the neutral thermal transit all run F x faster in plasma time; the gas then "
            "responds to the transit-scale fluctuations of S, which a physical gas cannot. F MUST NOT change the plasma plateau if the gas is quasi-static over the "
            "plasma time (both F = 1 and F = 10 deplete the channel-mean density by < 3 % over the run). The neutrals-spatial / neutrals-spatial-F10 pair is the "
            "predeclared QUALIFICATION of that statement (acceptance f_qualification): if the plateau scalars agree inside the particle band, F may be used in later "
            "runs to reach the neutral steady state; if not, F is disqualified and only F = 1 runs may be quoted"
            + ("" if f_acc == 1.0 else f"; THIS case is the F = {f_acc:g} member of the pair (72 us of neutral time over 3 transits = 1/3 of the tube residence)")),
        "wall_temperature_k": NEUTRAL_WALL_TEMPERATURE_K,
        "accommodation_coefficient": 1.0,
        "wall_recycling": True,
        "recombination_coefficient": 1.0,
        "wall_note": ("wall-ion recycling ON (recombination 1.0: every absorbed wall / anode ion returns as a thermal atom at T_w 500 K at the impact cell; full accommodation): "
                      "the ss-v4 template's v1.3 closure had NO recycling, so the R5 model is also the v1.4 recycling physics at the impact point. At F = 1 the recycled "
                      "inventory over the run is < 0.3 % of the channel atoms (wall + anode ion current ~2-3 mA = 1.5e16 atoms/s x 7 us); T_w 500 K is the v1.4 / R5 "
                      "shakedown convention (feed gas 300 K)"),
        "initial_profile": "knudsen",
        "initial_density_per_m3": None,
        "initial_profile_note": ("the free-molecular closed-end (Knudsen) profile scaled so that the exit-plane density is Q_in / c = 5.5e19 (the 0-D value): anode "
                                 f"{KNUDSEN_ANODE_DENSITY_PER_M3:.3g} m^-3, channel mean {KNUDSEN_CHANNEL_MEAN_PER_M3:.3g}; the R5 shakedown's window profile read anode 6.0e20 -> "
                                 "6 mm 4.1e20 -> 12 mm 2.5e20 -> 18 mm 1.1e20 -> exit 1.0e20 (axis, incl. the end correction)"),
        "max_ceiling_violation_fraction": 1.0e-3,
        "max_ceiling_violation_fraction_note": "declared tolerance of the clamp (see operating_point.neutral_density_role): the run fails closed above it",
        "metastables": {
            "model": "metastables_v1",
            "branching": [0.45, 0.35, 0.50, 0.35],
            "branching_note": ("metastable (6s[3/2]_2) share of the Biagi-v7.1 levels 8.315 / 9.447 / 9.917 / 11.7 eV (spec v2.5.0 derivation from the BSR level shares + the "
                               "Aymar-Coulombe 1978 6p cascade; net ~0.43, x3 uncertain); the level-resolved branching is why metastables_v1 REQUIRES xe_collision_set_v2"),
            "weight_ratio": 0.02,
            "ceiling_fraction": 0.05,
            "beb_kinetic_ev": None,
            "stepwise_scale": 1.0,
            "superelastic": True,
            "superelastic_level": 0,
            "superelastic_weight_ratio": 0.2,
            "wall_deexcitation_probability": 1.0,
            "radiative_decay_rate_per_s": 0.0,
            "metastables_note": ("Xe 6s[3/2]_2 pool as state 1 of the neutral arrays at weight_ratio x W_n; BEB stepwise ionisation (Kim and Rudd 1994; B = 3.815 eV; peak "
                                 "8.4e-20 m2 at 16.6 eV; stepwise_scale 1), superelastic de-excitation by Klein-Rosseland detailed balance from the bound level-1 table, "
                                 "wall de-excitation with probability 1, no radiative decay (trapped resonance levels are not pooled). R5 shakedown: channel-mean fraction "
                                 "0.27-0.33 % (up to 11 % locally), stepwise 3.1-3.4 % of the ionisation"),
        },
    }


# -- the six sealed cases ---------------------------------------------------------------------------------------------------------------
CASES: dict[str, dict[str, Any]] = {
    "coulomb": {
        "label": "Coulomb (R4)", "effects": ["coulomb_v1"], "see": False, "collision_set": False, "coulomb": True, "neutrals": False, "time_acceleration": None,
        "alpha": 0.0, "group": "single_effect",
        "role": "R4 alone: e-e + e-i Coulomb collisions (model v2.4.0) on the legacy lumped set, the 0-D inventory, the absorbing wall and alpha = 0 - the audit's gap (d) "
                "isolated against the recorded ss-v4 plateau",
    },
    "neutrals-spatial": {
        "label": "spatial neutrals + metastables (R5), F = 1", "effects": ["xe_collision_set_v2", "neutrals_spatial_v1", "metastables_v1"], "see": False, "collision_set": True,
        "coulomb": False, "neutrals": True, "time_acceleration": 1.0, "alpha": 0.0, "group": "single_effect",
        "role": "R5: the spatial (Knudsen-profile) neutral gas with the Xe(6s) metastable pool at PHYSICAL neutral time (F = 1), SEE off, Coulomb off, alpha = 0 - the "
                "OPERATING-POINT change (channel mean n_g ~2.5e20 vs the 0-D 5.5e19 -> 3.2e19 depleted) isolated against the recorded ss-v4 plateau. The metastable "
                "branching is level-resolved, so the case necessarily carries xe_collision_set_v2 (the runner refuses metastables_v1 on the legacy set): the R5-alone "
                "shift is read against the physics-effects xe-set-v2 record (pic2d_physics_effects_v1/results/xe-set-v2) as the SECONDARY reference once it exists; "
                "against ss-v4 the set-v2 contribution is predeclared inside the band on every plateau scalar (physics-effects hypotheses R3a)",
    },
    "neutrals-spatial-F10": {
        "label": "spatial neutrals + metastables (R5), F = 10", "effects": ["xe_collision_set_v2", "neutrals_spatial_v1", "metastables_v1"], "see": False, "collision_set": True,
        "coulomb": False, "neutrals": True, "time_acceleration": F_QUALIFICATION_FACTOR, "alpha": 0.0, "group": "f_qualification",
        "role": "the TIME-ACCELERATION qualification member: identical to neutrals-spatial except neutrals.time_acceleration = 10 (a different configuration identity); "
                "acceptance f_qualification = every plateau scalar agrees with the F = 1 member inside the particle band -> F qualified for later runs, else disqualified",
    },
    "full-physics-alpha0": {
        "label": "full physics, alpha = 0", "effects": ["see_dielectric_v1", "xe_collision_set_v2", "coulomb_v1", "neutrals_spatial_v1", "metastables_v1"], "see": True,
        "collision_set": True, "coulomb": True, "neutrals": True, "time_acceleration": 1.0, "alpha": 0.0, "group": "full_physics",
        "role": "every R2-R5 effect together (SEE from the BN wall, the collision set v2, Coulomb, the spatial gas with metastables at F = 1) with NO anomalous transport: "
                "the alpha = 0 point of the full model, the additivity statement's combined case (against see-bn+xe-set-v2 + coulomb + R5) and the reference of the "
                "full-physics alpha trend",
    },
    "full-physics-alpha1over16": {
        "label": "full physics, alpha = 1/16", "effects": ["bohm_perpendicular_rotation", "see_dielectric_v1", "xe_collision_set_v2", "coulomb_v1", "neutrals_spatial_v1", "metastables_v1"],
        "see": True, "collision_set": True, "coulomb": True, "neutrals": True, "time_acceleration": 1.0, "alpha": float(BOHM_ALPHA_SERIES[1]), "group": "full_physics",
        "role": "the full model with the classical Bohm closure nu_an = omega_ce / 16 (rotation model): at the dilute 0-D gas this alpha EXTINGUISHED the discharge "
                "(alpha-series launch 1, 0916a4f8: N_e e-fold 0.88 us = r_w^2 / 4 D_perp, I_d 3.1 -> 0.06 mA); the key hypothesis is that the Knudsen-profile gas (with SEE) "
                "SUSTAINS it (ignition gate passes, plateau reached)",
    },
    "full-physics-alpha0.345": {
        "label": "full physics, alpha = 0.345", "effects": ["bohm_perpendicular_rotation", "see_dielectric_v1", "xe_collision_set_v2", "coulomb_v1", "neutrals_spatial_v1", "metastables_v1"],
        "see": True, "collision_set": True, "coulomb": True, "neutrals": True, "time_acceleration": 1.0, "alpha": float(BOHM_ALPHA_SERIES[2]), "group": "full_physics",
        "role": "the full model at Brandt et al. 2016's coefficient (D_perp = 0.4 kT_e / eB as nu_an = 0.4 omega_ce -> exact Green-Kubo factor 0.345; the strongest leak): "
                "the ext-val bohm-0.4 launch 2 at Brandt's static 2e20 gas sustained a marginal discharge past 1.2 transits; does the Knudsen gas of THIS device (channel "
                "mean ~2.5e20, anode 5.5e20, exit 7e19) sustain it? Launched FIRST (the sustain question decides the value of the rest of the campaign)",
    },
}
LAUNCH_PRIORITY: tuple[str, ...] = ("full-physics-alpha0.345", "full-physics-alpha0", "neutrals-spatial", "full-physics-alpha1over16", "coulomb", "neutrals-spatial-F10")
FULL_PHYSICS_CASES: tuple[str, ...] = ("full-physics-alpha0", "full-physics-alpha1over16", "full-physics-alpha0.345")
F_PAIR: tuple[str, str] = ("neutrals-spatial", "neutrals-spatial-F10")
SPATIAL_CASES: tuple[str, ...] = tuple(c for c, m in CASES.items() if m["neutrals"])

# -- the shift table ---------------------------------------------------------------------------------------------------------------------
# the physics-effects quantities (relative shifts with the ss-v4 particle band; absolute for the IEDF fraction) + this campaign's readings
QUANTITY_KEYS: tuple[str, ...] = (
    "discharge_current_a", "exit_ion_beam_a", "ionization_rate_per_s", "gross_utilisation", "neutral_density_per_m3", "peak_n_e_window_per_m3", "t_e_peak_window_ev",
    "iedf_low_energy_fraction", "anode_ion_a", "wall_electron_power_w", "wall_ion_mean_energy_ev",
    "ionization_centroid_z_m", "neutral_density_anode_over_exit", "neutral_depletion_fraction", "metastable_fraction_of_ground", "stepwise_fraction_of_ionization",
    "nu_e_spitzer_peak_over_nu_en",
)
PLATEAU_SCALARS: tuple[str, ...] = ("discharge_current_a", "exit_ion_beam_a", "ionization_rate_per_s", "gross_utilisation", "neutral_density_per_m3", "peak_n_e_window_per_m3",
                                    "t_e_peak_window_ev")
ABSOLUTE_BAND: dict[str, float] = {**PE_ABSOLUTE_BAND, "ionization_centroid_z_m": CENTROID_BAND_M}
# for the spatial cases the channel-mean n_g shift vs ss-v4 is the Knudsen profile BY CONSTRUCTION (+680 %): reported, never judged
REPORTED_ONLY_SPATIAL: tuple[str, ...] = ("neutral_density_per_m3",)

# -- predeclared hypotheses (the SIGN is the hypothesis; the magnitude is what the run measures) ---------------------------------------------
HYPOTHESES_COULOMB: dict[str, dict[str, Any]] = {
    "ionization_rate_per_s": {"sign": "+", "expected": "+5 to +20 % (audit section 4.d)", "reason": "e-e collisions refill the ionising tail the inelastic losses drain "
                              "(a Maxwellianising operator raises the rate coefficient of a depleted-tail distribution)"},
    "gross_utilisation": {"sign": "+", "expected": "follows S", "reason": "utilisation = S / feed"},
    "exit_ion_beam_a": {"sign": "+", "expected": "follows S (weak)", "reason": "more ions born, the beam fraction unchanged"},
    "discharge_current_a": {"sign": "0", "expected": "unchanged inside the 5.7 % band", "reason": "e-i drag at nu_ei ~ 1e-2 nu_en carries no anode current; the anode "
                            "electron flux is set by the cusp transport"},
    "t_e_peak_window_ev": {"sign": "-", "expected": "-3 to -8 % (may sit inside the 9.3 % band)", "reason": "the hot mirror-trapped tail is thermalised toward the bulk"},
    "peak_n_e_window_per_m3": {"sign": "+", "expected": "follows S (weak; may sit inside the 11.9 % band)", "reason": "more ionisation at the same confinement"},
    "neutral_density_per_m3": {"sign": "-", "expected": "-2 to -6 % (may sit inside the 4 % band)", "reason": "more ionisation depletes the 0-D inventory further"},
    "nu_e_spitzer_peak_over_nu_en": {"sign": "+", "expected": "0.15-0.4 at the plateau peak (reference 0 by construction)", "reason": "the audit's gap-(d) estimate at 1e18 / 5 eV"},
}
HYPOTHESES_R5: dict[str, dict[str, Any]] = {
    "ionization_rate_per_s": {"sign": "+", "expected": "x2-4 (the channel mean n_g is 4.5 x the 0-D value; bounded by the electron inventory)",
                              "reason": "nu_iz is proportional to the local gas density and the gas is densest where the electrons are (the anode-side flames)"},
    "gross_utilisation": {"sign": "+", "expected": "up; MAY EXCEED 1 at F = 1", "reason": "S / Q_in with a quasi-frozen gas that does not deplete over the run: a gross "
                          "utilisation > 1 is the recorded signature that the neutral steady state was not reached (only F >> 1 reaches it), not a physical yield"},
    "discharge_current_a": {"sign": "+", "expected": "+30 to +100 %", "reason": "every ionisation adds an electron that leaves through the anode or the wall; the CEX-slowed "
                            "ion population raises the anode ion current too"},
    "exit_ion_beam_a": {"sign": "+", "expected": "up with S", "reason": "more ions born in the channel; part reach the exit"},
    "peak_n_e_window_per_m3": {"sign": "+", "expected": "+20 to +100 %", "reason": "the ionisation source rises faster than the cusp losses; the peak may approach the "
                               "33 um grid's Debye limit (pi cells per lambda_D at ~2.7e18 / 5.6 eV) - a peak-Debye stop is a recorded resolution outcome"},
    "t_e_peak_window_ev": {"sign": "-", "expected": "-10 to -30 %", "reason": "the denser gas cools the tail by inelastic collisions (plus the per-event loss of the four levels)"},
    "ionization_centroid_z_m": {"sign": "-", "expected": "-1 to -4 mm (upstream; band 1 mm)", "reason": "the gas is 8 x denser at the anode than at the exit: the flames "
                                "move toward the anode-side cusps (the R5 shakedown's seed-transient centroid 13.2 mm was not like-for-like)"},
    "iedf_low_energy_fraction": {"sign": "+", "expected": "+0.15 to +0.40 absolute", "reason": "CEX at the denser gas: lambda_CEX ~ 15 mm at 2.5e20 vs 12-24 mm flight paths"},
    "anode_ion_a": {"sign": "+", "expected": "up", "reason": "slow CEX ions born upstream fall to the anode"},
    "neutral_density_per_m3": {"sign": "+", "expected": "+680 % BY CONSTRUCTION (Knudsen channel mean 2.5e20 vs the depleted 0-D 3.2e19): reported, not judged",
                               "reason": "the operating-point change itself"},
    "neutral_density_anode_over_exit": {"sign": "+", "expected": "5-10 (the Knudsen ratio incl. the exit end correction; reference 1 by construction); read on the window-mean "
                                        "density averaged over the inner third of the radius (the single axis cell holds 0.4-3 macro-neutrals and is shot noise)",
                                        "reason": "free-molecular closed-end profile"},
    "neutral_depletion_fraction": {"sign": "+", "expected": "< 0.01 at F = 1 (~0.03 at F = 10)", "reason": "quasi-frozen gas over 3 transits"},
    "metastable_fraction_of_ground": {"sign": "+", "expected": "0.2-0.5 % channel mean at a filled pool (reference 0 by construction; the R5 shakedown's 0.27-0.33 % was at F = 100). "
                                      "The pool is QUANTISED at weight_ratio x W_n = 4.4e5 atoms per macro-metastable: at F = 1 a cell produces ~50 pool atoms per sub-step, so "
                                      "its first macro-metastable spawns after ~2 us and the pool is still filling (and coarsely resolved: ~3 spawns per cell over 3 transits) at "
                                      "the plateau - the F = 1 fraction reads LOW by construction; at F = 10 the pool fills in ~1 us of plasma time (this is one of the "
                                      "predeclared F distortions, reported not judged)", "reason": "R5 shakedown 0.27-0.33 % at F = 100"},
    "stepwise_fraction_of_ionization": {"sign": "+", "expected": "3-6 % of S (reference 0 by construction)", "reason": "BEB stepwise ionisation from the 6s pool; R5 shakedown 3.1-3.4 %"},
}
HYPOTHESES_FULL_ALPHA0: dict[str, dict[str, Any]] = {
    **HYPOTHESES_R5,
    **{k: v for k, v in HYPOTHESES_COULOMB.items() if k in ("nu_e_spitzer_peak_over_nu_en",)},
    "discharge_current_a": {"sign": "+", "expected": "+50 to +150 % (SEE +10-30 % on top of the operating-point change)", "reason": HYPOTHESES_R5["discharge_current_a"]["reason"] +
                            "; the SEE-lowered cusp sheaths admit more primaries"},
    "peak_n_e_window_per_m3": {"sign": "+", "expected": "up (the gas dominates: R5 +20-100 % against SEE's -5 to -15 %)", "reason": "operating point over the SEE de-confinement"},
    "t_e_peak_window_ev": {"sign": "-", "expected": "-20 to -40 % (all three cool: SEE secondaries at 2 eV, the per-event loss, the denser gas)", "reason": "R2 + R3a + R5"},
    "wall_electron_power_w": {"sign": "+", "expected": "up (x1.5-2 from SEE; more from the denser plasma)", "reason": "the lower sheath admits more primaries"},
    "wall_ion_mean_energy_ev": {"sign": "-", "expected": "down by the sheath fraction", "reason": "ions fall through a smaller (SEE) sheath drop"},
    "cusp_sheath_drop_v": {"sign": "-", "expected": "-10 to -45 % (SEE; per cusp beside the effective yield and the SCL flag)", "reason": "the emitted current lowers the floating-wall drop"},
}
# the alpha cases: the KEY hypothesis is 'sustains'; the sign rows are the alpha-trend signs judged against full-physics-alpha0 (secondary)
HYPOTHESES_FULL_ALPHA: dict[str, dict[str, Any]] = {
    "sustains": {"sign": "sustains", "expected": "ignition gate passes at 1.0 and 2.0 us AND the plateau is reached",
                 "reason": "a Bohm-leaky discharge extinguished at the dilute 0-D gas (n_g 5.5e19; e-fold r_w^2 / 4 D_perp) and sustained at Brandt's 2e20: the Knudsen gas "
                           "(channel mean 2.5e20, anode 5.5e20) with SEE-lowered sheaths should supply the ionisation the leak removes"},
    **{k: {"sign": v["sign"], "expected": f"{v['expected_at_1over16']} at 1/16 (the alpha-series' expectation at the 0-D gas; the magnitude at the Knudsen gas is the measurement)",
           "reason": v["reason"], "judged_against": "full-physics-alpha0"} for k, v in at_protocol.HYPOTHESES.items()
      if k in ("discharge_current_a", "ionization_rate_per_s", "gross_utilisation", "peak_n_e_window_per_m3", "t_e_peak_window_ev", "exit_ion_beam_a",
               "cusp_electron_wall_current_a", "cusp_sheath_drop_v")},
    "neutral_density_per_m3": {"sign": "0", "expected": "unchanged inside the 4 % band (the gas is quasi-frozen at F = 1: it CANNOT respond to the leak over 3 transits)",
                               "reason": "the alpha-series' 'n_g up' is a fixed-point statement the frozen gas cannot make", "judged_against": "full-physics-alpha0"},
}
HYPOTHESES_BY_CASE: dict[str, dict[str, dict[str, Any]]] = {
    "coulomb": HYPOTHESES_COULOMB, "neutrals-spatial": HYPOTHESES_R5, "neutrals-spatial-F10": HYPOTHESES_R5, "full-physics-alpha0": HYPOTHESES_FULL_ALPHA0,
    "full-physics-alpha1over16": HYPOTHESES_FULL_ALPHA, "full-physics-alpha0.345": HYPOTHESES_FULL_ALPHA,
}
KEY_QUANTITIES: dict[str, tuple[str, ...]] = {
    "coulomb": ("ionization_rate_per_s",),
    "neutrals-spatial": ("ionization_rate_per_s", "discharge_current_a"),
    "neutrals-spatial-F10": ("ionization_rate_per_s", "discharge_current_a"),
    "full-physics-alpha0": ("discharge_current_a", "ionization_rate_per_s", "t_e_peak_window_ev"),
    "full-physics-alpha1over16": ("sustains",),
    "full-physics-alpha0.345": ("sustains",),
}
# alpha-trend monotone set at full physics (the alpha-series' MONOTONE_QUANTITIES minus n_g, which the frozen gas cannot move)
MONOTONE_QUANTITIES: tuple[str, ...] = ("discharge_current_a", "ionization_rate_per_s", "gross_utilisation", "peak_n_e_window_per_m3", "t_e_peak_window_ev")


# -- reference block -------------------------------------------------------------------------------------------------------------------
def ionization_centroid_from_maps(maps: dict[str, np.ndarray], grid) -> dict[str, Any] | None:
    """Axial centroid / quartiles of the node ionisation-rate map (volume-weighted by the node volume through the deposit itself)."""

    if "ionization_rate_per_m3_s" not in maps:
        return None
    node_map = np.nan_to_num(np.asarray(maps["ionization_rate_per_m3_s"], dtype=np.float64))
    # node volumes: 2 pi r dr dz with the axis node's r -> dr / 4 (the CIC axis correction); the ratio is all that matters
    nr, nz = node_map.shape
    r = np.arange(nr) * grid.dr_m
    r[0] = 0.25 * grid.dr_m
    weight = (2.0 * np.pi * r * grid.dr_m * grid.dz_m)[:, None] * node_map
    profile = weight.sum(axis=0)
    total = float(profile.sum())
    if total <= 0.0:
        return {"total": 0.0}
    z = grid.geometry.z_min_m + np.arange(nz) * grid.dz_m
    cdf = np.cumsum(profile) / total
    quartile = lambda q: float(z[min(int(np.searchsorted(cdf, q)), z.size - 1)])  # noqa: E731
    return {"total_weighted": total, "centroid_z_m": float(np.sum(profile * z) / total), "z25_m": quartile(0.25), "z50_m": quartile(0.5), "z75_m": quartile(0.75),
            "fraction_upstream_of_12mm": float(cdf[min(int(np.searchsorted(z, 0.012)), z.size - 1)])}


def reference_extras_from_v4(results: Path = V4_RESULTS) -> dict[str, Any] | None:
    """The ss-v4 readings this campaign adds to the physics-effects reference block (ionisation centroid; the by-construction zeros / ones of the 0-D gas)."""

    if not (results / "maps.npz").is_file() or not (results / "summary.json").is_file():
        return None
    from experiments.pic2d_cft_steady_state_v1 import run as runner  # local: heavy

    with np.load(results / "maps.npz") as archive:
        maps = {k: np.asarray(archive[k]) for k in archive.files}
    grid = runner.build_config(load_v4_protocol(), backend="cpu").grid
    centroid = ionization_centroid_from_maps(maps, grid)
    return {
        "ionization_centroid_z_m": None if centroid is None else centroid["centroid_z_m"],
        "ionization_centroid_detail": centroid,
        # the 0-D inventory is uniform, has no metastables and no Coulomb operator: these read exactly 1 / 0 / 0 / 0 / 0 by construction
        "neutral_density_anode_over_exit": 1.0, "neutral_depletion_fraction": None, "metastable_fraction_of_ground": 0.0, "stepwise_fraction_of_ionization": 0.0,
        "nu_e_spitzer_peak_over_nu_en": 0.0,
    }


def v4_reference_block() -> dict[str, Any]:
    """The reference point of every case: the recorded ss-v4 plateau (0d228ad2; SEE off, legacy set, 0-D gas, no Coulomb, alpha = 0) with its CORRECTED ledger status."""

    block = pe_protocol.v4_reference_block()
    extras = reference_extras_from_v4()
    if extras is not None:
        block["quantities"].update({k: v for k, v in extras.items() if k != "ionization_centroid_detail"})
        block["ionization_centroid_detail"] = extras["ionization_centroid_detail"]
    block["quantities_added_by_this_campaign"] = list(block.get("quantities_added_by_this_campaign", [])) + [
        "ionization_centroid_z_m (node-volume-weighted axial centroid of maps.npz ionization_rate_per_m3_s)",
        "neutral_density_anode_over_exit = 1, metastable_fraction_of_ground = 0, stepwise_fraction_of_ionization = 0, nu_e_spitzer_peak_over_nu_en = 0 (by construction of the 0-D / "
        "collisionless / legacy reference); neutral_depletion_fraction has no reference (the 0-D density IS the depleted fixed point)",
    ]
    block["neutral_model"] = "0-D quasi-steady inventory (v1.3 closure, no recycling): uniform n_g = 3.19e19 at the plateau = the EXIT density of the Knudsen profile at the same feed"
    block["absolute_band"] = ABSOLUTE_BAND
    block["particle_band_note"] = (block["particle_band_note"] + "; the ionisation centroid has NO replicate band - the declared absolute detection band "
                                   f"{CENTROID_BAND_M * 1e3:.0f} mm (30 cells) is a preregistered choice; the spatial cases' channel-mean n_g shift is BY CONSTRUCTION and is reported, never judged")
    block["secondary_reference"] = {
        "for_cases": list(SPATIAL_CASES), "experiment": "modern/experiments/pic2d_physics_effects_v1", "case": "xe-set-v2",
        "results_dir": "modern/experiments/pic2d_physics_effects_v1/results/xe-set-v2 (assessment.json; results-only commit when that campaign's launch 2 finishes)",
        "role": "the R5 cases carry xe_collision_set_v2 by necessity (metastables_v1); the R5-ALONE shift = shift(neutrals-spatial) - shift(xe-set-v2) once that record exists; "
                "until then the set-v2 contribution is the physics-effects R3a hypothesis (I_d 0, S / T_e -3 to -5 %: inside the band on every plateau scalar) and the R5 "
                "shift vs ss-v4 is reported as the combined R3 + R5 shift",
    }
    return block


# -- composition ------------------------------------------------------------------------------------------------------------------------
def _anomalous_block(alpha: float) -> dict[str, Any]:
    return {
        "model": ANOMALOUS_MODEL_ROTATION, "alpha": alpha,
        "alpha_note": (f"model v2.1.0 Bohm-type anomalous transport (the alpha-series' block, 33be2a89): every electron has its velocity rotated about the local B by a "
                       f"uniform random angle (v_parallel and |v| unchanged, gyro-centre shifted - Brandt et al. 2016's event model) with probability 1 - exp(-alpha omega_ce dt) "
                       f"per step; nu_an = {alpha:.6g} omega_ce = {alpha * 1.7588e11 * 0.05:.3g} s^-1 at 0.05 T, {alpha * 1.7588e11 * 0.2914:.3g} s^-1 at the channel max |B| 0.2914 T; "
                       f"D_perp = (kT_e/eB) alpha/(1+alpha^2) = {alpha / (1 + alpha**2):.4f} kT_e/eB. Elastic, tallied in cumulative.anomalous; a separate exact-Poisson process "
                       f"outside the MCC null budget (alpha = 0 without the block is bitwise). At the dilute 0-D gas alpha = 1/16 EXTINGUISHED the discharge in 1 transit "
                       f"(e-fold 0.88 us = r_w^2 / 4 D_perp); here the same closure meets the Knudsen gas and the SEE wall"),
    }


def compose_case_protocol(case_id: str, *, wall_budget_seconds: float | None = None, budget_note: str | None = None) -> dict[str, Any]:
    """The ss-v4 protocol with this campaign's changes for ``case_id`` (deterministic; sealed under ``protocols/``)."""

    if case_id not in CASES:
        raise KeyError(f"unknown case {case_id!r}; cases: {sorted(CASES)}")
    case = CASES[case_id]
    alpha = float(case["alpha"])
    p = copy.deepcopy(load_v4_protocol())
    p["schema_version"] = CASE_SCHEMA_VERSION
    p["experiment_id"] = f"{EXPERIMENT_ID}-{case_id}"
    changes = ["numerics.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak (v2.0.6 floor)", "numerics.performance.moment_sample_interval = 5 (v2.0.5)",
               "stopping_rule.grid_heating_triad.drift_members_arming (model v2.1.1 latch) + stopping_rule.ignition_gate (v2.0 gate; both from the alpha-series amendment 1)",
               "stopping_rule.wall_budget_seconds (launch-box measured rate x 1.5)",
               "stopping_rule.acceptance (a plateau, b corrected ledger, c shift table + per-cusp / effect readings, d verdict incl. extinguished, e sustain, f F-qualification, g additivity)",
               "case.id", "status / classification / model_version / model_spec / claim_boundary / simplifications text",
               "reference_run -> the ss-v4 plateau with its corrected-ledger status (+ this campaign's added readings)"]
    if case["neutrals"]:
        changes.insert(0, "operating_point.neutrals (model v2.5.0 neutrals_spatial_v1 + metastables_v1, F = %g) REPLACES operating_point.neutral_inventory; "
                          "operating_point.neutral_density_per_m3 -> the MCC ceiling %.3g (above the Knudsen anode density)" % (case["time_acceleration"], MCC_CEILING_SPATIAL_PER_M3))
    if case["coulomb"]:
        changes.insert(0, "numerics.coulomb (model v2.4.0 coulomb_v1: e-e + e-i, cycle 10, i-i off)")
    if case["see"]:
        changes.insert(0, "numerics.see (model v2.2.0 see_dielectric_v1, BN preset)")
    if case["collision_set"]:
        changes.insert(0, "operating_point.collision_set (model v2.3.0 xe_collision_set_v2, ion_neutral true) + the documentary cross_sections entry")
    if alpha > 0.0:
        changes.insert(0, "numerics.anomalous_collisions (model bohm_perpendicular_rotation, alpha)")
    p["campaign"] = {"experiment_id": EXPERIMENT_ID, "case": case_id, "label": case["label"], "group": case["group"], "effects": list(case["effects"]),
                     "see": bool(case["see"]), "collision_set": bool(case["collision_set"]), "coulomb": bool(case["coulomb"]), "neutrals_spatial": bool(case["neutrals"]),
                     "time_acceleration": case["time_acceleration"], "alpha": alpha, "role": case["role"], "launch_priority": list(LAUNCH_PRIORITY),
                     "template": "modern/experiments/pic2d_cft_steady_state_v4/protocol.json (every key not named in campaign.changes is byte-for-byte its value)",
                     "changes": changes,
                     "companion_campaigns": {
                         "alpha_series_at_the_0d_gas": "modern/experiments/pic2d_anomalous_transport_v1 (33be2a89 amendment 1): alpha = 1/16 EXTINGUISHED (record 0916a4f8); 1/64 and 0.345 queued",
                         "physics_effects_r2_r3": "modern/experiments/pic2d_physics_effects_v1 (79a7c87a): see-bn / xe-set-v2 / see-bn+xe-set-v2 queued - the single-effect parts of the additivity statement",
                         "ext_val_bohm_0_4": "modern/experiments/pic2d_external_validation_v0 launch 2 (a1065ce4): alpha 0.345 at Brandt's static 2e20 gas",
                     }}
    p["status"] = "preregistered_full_physics_r4_r5_combined_not_validated"
    p["classification"] = ("axisymmetric_electrostatic_pic_mcc_full_physics_" + "_".join(e for e in case["effects"]) +
                           f"_alpha_{case_id.split('alpha')[-1] if alpha > 0 else '0'}_on_the_33um_reference_plateau_v2_0_6_gates_v2_1_1_arming_not_validated")
    p["model_version"] = MODEL_VERSION
    p["model_spec"] = MODEL_SPEC
    num = p["numerics"]
    op = p["operating_point"]
    if alpha > 0.0:
        num["anomalous_collisions"] = _anomalous_block(alpha)
    if case["see"]:
        num["see"] = pe_protocol._see_block()
    if case["coulomb"]:
        num["coulomb"] = {**copy.deepcopy(COULOMB_BLOCK), "coulomb_note": COULOMB_NOTE}
    if case["collision_set"]:
        op["collision_set"] = pe_protocol._collision_set_block()
        p["cross_sections"] = {"electron": f"modern/spec/pic2d/{pe_protocol.XE_ELECTRON_SET_V2_FILE} (payload sha256 {pe_protocol.XE_ELECTRON_SET_V2_PAYLOAD_SHA256})",
                               "ion_neutral": f"modern/spec/pic2d/{pe_protocol.XE_ION_NEUTRAL_SET_V1_FILE} (payload sha256 {pe_protocol.XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256})",
                               "note": "documentary: the runner loads the electron set named by operating_point.collision_set and hash-checks it; the legacy v1 file (the "
                                       "reference's) is modern/spec/pic2d/xenon-cross-sections-v1.json"}
    if case["neutrals"]:
        removed = op.pop("neutral_inventory")
        op["neutrals"] = neutrals_block(float(case["time_acceleration"]))
        op["neutral_density_per_m3"] = MCC_CEILING_SPATIAL_PER_M3
        op["neutral_density_role"] = MCC_CEILING_NOTE
        op["neutral_model_note"] = ("v2.5.0: operating_point.neutrals (neutrals_spatial_v1 + metastables_v1) REPLACES operating_point.neutral_inventory of the ss-v4 protocol; the "
                                    f"removed 0-D block was feed {removed['feed_atoms_per_s']:.6g} /s (kept as the spatial feed), tau_g {removed['relaxation_time_s']:.3g} s (the "
                                    "artificial 0-D relaxation; the spatial model has the declared time acceleration instead), no wall recycling (v1.3; the spatial model recycles "
                                    "at the impact cell)")
        num["neutral_update_note"] = ("v2.5.0: the neutral gas is a test-particle population marched every substep_steps = 200 steps (the device-sync interval) on its own "
                                      "RNG stream (CPU stream 6 / seed-table column 6 in the v2.5.0 layout); its nearest-cell density is published to a device-resident "
                                      "per-cell array read by both MCCs (never a kernel scalar: the CUDA graph replays it)")
    gate = num["peak_debye_gate"]
    gate["min_accumulated_macro_particle_steps_at_peak"] = 64000
    gate["min_accumulated_macro_particle_steps_at_peak_note"] = (
        "v2.0.6 (spec gates_v2_0.peak_debye_gate_accumulated_floor_v2_0_6): the gated node is the densest node whose ACCUMULATED macro-electron-steps over the "
        "400 000-step window reach 64 000 (= 32 crossings x ~2000 steps), so axis columns that hold < 32 macro-electrons per step but are visited by many are "
        "gate-able (the v2.0.3 mean-occupancy floor of 32 made them invisible); on the v4 maps the gated statistic is unchanged (2.154 at the same node, "
        "resolved nodes 19 650 -> 42 130); the mean-occupancy floor stays recorded alongside. At the denser Knudsen gas the peak n_e is EXPECTED to rise: the hard "
        "gate (pi cells per lambda_D ~ 2.7e18 at 5.6 eV) may stop a spatial case - a recorded resolution outcome (no_plateau, stop reason peak_debye_gate), not a failure of the physics")
    num["performance"] = {"moment_sample_interval": 5,
                          "moment_sample_interval_note": "v2.0.5: electron window moments (T_e maps, peak-Debye T_e, sample counts) sampled every 5th accumulated step; "
                                                         "physics bitwise, gated Delta/lambda_D moves by 1.7e-5 median vs K = 1 (8aca6c3a); enters config_sha256 by the "
                                                         "v2.0.5 identity policy (K != 1 is a declared configuration); the reference ran K = 1"}
    num["frame_recorder_note"] = ("v2.0 frame recorder ON (28 ns frames): the ionisation / density / potential structure at the cusp planes against the v4 frames (0d228ad2); "
                                  "with SEE the frames gain wall_see_*; with the spatial gas the neutral / metastable cell sums ride the frame differences; frames are "
                                  "diagnostics, not gates")
    f_tag = "" if not case["neutrals"] else f"-F{case['time_acceleration']:g}"
    gas_tag = f"knudsen-gas-ceiling-{MCC_CEILING_SPATIAL_PER_M3:.2g}{f_tag}" if case["neutrals"] else "ng0-5.5e19-inventory-v1.3-closure"
    p["case"]["id"] = f"{case_id}-33um-dt1.4ps-w2.667e4-{gas_tag}-seed5e16"
    p["case"]["seed_note"] = ("seed 20260903 = the ss-v4 seed: every new process draws from its own RNG stream (anomalous: stream 3 / seed-table slot 2; ion MCC: column 3; "
                              "SEE: column 4 / CPU stream 5; Coulomb: column 5 / stream 6; neutrals: column 6 / stream 6 of the v2.5.0 layout), so the electron-MCC and "
                              "injection draws of the reference are reproduced step for step until the first new event")
    p["budget_v1_3"]["cost_model"] = {
        "source": ("the launch-box (Lambda H100, CUDA MPS, 4 PIC clients) measurements: alpha-series preflights 4.77 ms/step at 4.5 M particles (16:48 UTC 2026-09-04); "
                   "physics-effects preflights 6.33 (SEE) / 5.60 (set v2) / 6.52 (both) as the 5th client; R4 shakedown +0.48 ms/step amortised for the Coulomb cycle; R5 "
                   "shakedown 4.9-5.4 vs 4.52 ms/step (the neutral sub-step ~+10 %)"),
        "steps_to_3_transits": STEPS_TO_3_TRANSITS,
        "a_priori_hours_to_3_transits": "6-8 ms/step in an MPS slot -> 8.6-11.4 h (the full-physics cases at the upper end; a denser plateau costs ~1 ms/step per extra M particles)",
        "gpu_memory_estimate_gb": "~2-3 (the v4 device pool + the SEE reservation + the Coulomb permutation buffers + ~0.5 GB of neutral particles at 4 M macro-neutrals)",
        "budget_basis": "wall_budget_seconds = 1.5 x the launch-box measured plateau-load rate (the LARGER of the 4.5 M and the 7.7 M-particle timings for the spatial cases, "
                        "whose peak density is expected to rise) x steps_to_3_transits, recorded in preflight-<case>.json before the preregistration commit",
    }
    stop = p["stopping_rule"]
    if wall_budget_seconds is not None:
        stop["wall_budget_seconds"] = float(wall_budget_seconds)
        stop["wall_budget_note"] = budget_note or "1.5 x the launch-box measured plateau-load rate x 3-transit steps (preflight-<case>.json)"
    else:
        stop["wall_budget_seconds"] = 54000.0
        stop["wall_budget_note"] = ("A-PRIORI 15.0 h = 1.5 x 10.0 h (7 ms/step in an MPS slot x 5.14 M steps); REPLACED by the launch-box measured rate x 1.5 before the "
                                    "preregistration commit (compose --budget-from-preflight)")
    stop["fail_closed"] = stop["fail_closed"].replace("v2.0.3 window-mode peak-node Debye gate", "v2.0.6 window-mode peak-node Debye gate (accumulated-particle-step floor)") \
        .replace("v2.0.3 windowed residual-power bound", "v2.0.3 windowed residual-power bound on the v2.0.6 W-corrected ledger")
    stop["fail_closed"] += ("; model v2.1.1: the triad's drift members are armed by the settled-once latch (grid_heating_triad.drift_members_arming: >= 2 transits AND the I_d "
                            "drift has read < 5 % at a checkpoint) instead of at 1.0 transit, and the v2.0 ignition gate (stopping_rule.ignition_gate) stops an extinguished "
                            "discharge at 1.0 / 2.0 us (stop_reason no_ignition = the recorded outcome 'extinguished')")
    if case["collision_set"]:
        stop["fail_closed"] += "; v2.3.0: an ion-MCC null-collision ceiling violation ends the run at the next series record (PIC2DStabilityError)"
    if case["see"]:
        stop["fail_closed"] += "; v2.2.0: an overflow of the Warp SEE birth reservation (256 + N_e / 1000 per step) fails closed at the sync"
    if case["neutrals"]:
        stop["fail_closed"] += ("; v2.5.0: a neutral-particle capacity overflow, an unresolved neutral flight, an atom-ledger identity failure or a published density above the "
                                f"MCC ceiling {MCC_CEILING_SPATIAL_PER_M3:.3g} in more than 1e-3 of an interval's plasma cell-substeps fails closed at the sync")
    stop["grid_heating_triad"]["note"] += ("; v2.0.6: the ledger's inelastic_loss_j carries W, so the windowed statistic IS the corrected one (the recorded ss-v4 series read -7.7 % where "
                                           "the corrected value was +2.46 %); the new sinks / sources (ion_neutral_loss_j, ke_see_emitted_j, ke_coulomb_j, the metastable "
                                           "+(E_iz - E_m) / -E_m terms) are booked, so the residual stays the numerical-heating witness"
                                           "; model v2.1.1: enforced_after_transit_times 1.0 is SUPERSEDED by drift_members_arming (kept as the recorded v1.4 rule); the "
                                           "residual-power member is unchanged")
    stop["grid_heating_triad"]["drift_members_arming"] = copy.deepcopy(DRIFT_MEMBERS_ARMING)
    stop["ignition_gate"] = copy.deepcopy(IGNITION_GATE)
    stop["ignition_check"] = ("the seed and injection are the ss-v4 ones; NO adjustment is allowed under this preregistration - a non-ignition / extinction is a recorded outcome of "
                              "the closure at this operating point (stop_reason no_ignition -> plateau_status 'extinguished'), never a reason to relaunch")
    hypotheses = HYPOTHESES_BY_CASE[case_id]
    reported_only = list(REPORTED_ONLY_SPATIAL) if case["neutrals"] else []
    stop["acceptance"] = {
        "declared": "predeclared before the launch; evaluated by `run.py assess --case <case>` (per case) and `run.py assess --campaign` (sustain table, alpha trend, additivity, "
                    "F qualification) against reference_run (the recorded ss-v4 plateau) and the companion records; verdicts recorded in results/<case>/assessment.json and "
                    "results/campaign-assessment.json (results-only commits)",
        "a_plateau": "stop_reason == plateau_reached_after_min_transit_times under the v4 rule (>= 3 transits = 5 142 858 steps, trailing-20 % drifts of I_d, N_e, n_g < 5 %, triad "
                     "soft bounds, window-mode peak-Debye soft margin 2.5); the drift members of the triad arm by the v2.1.1 latch",
        "b_residual_power": "summary.grid_heating_triad.windowed_energy_residual_over_electrode_work (trailing 400 000-step ratio at the stop, v2.0.6 W-corrected ledger) < +0.02, "
                            "one-sided; the reference reads +0.0246 on the corrected ledger (FAIL) - a case that passes (b) is a cleaner plateau than the reference",
        "c_shifts": {
            "quantities": list(QUANTITY_KEYS),
            "rule": "for every quantity the shift (case - reference) is reported: relative for the plateau quantities with the ss-v4 particle band, ABSOLUTE for the IEDF "
                    "low-energy fraction (band 0.03) and the ionisation centroid (band 1 mm). A shift with a declared '+' / '-' hypothesis counts as CONFIRMING when it has the "
                    "declared sign AND exceeds the band, CONTRADICTING when it has the opposite sign AND exceeds the band, INSIDE THE BAND otherwise; a '0' hypothesis is "
                    "CONFIRMING inside the band and CONTRADICTING beyond it; quantities without a band and the reported_only quantities are REPORTED with their hypothesis "
                    "sign and never judged. For the full-physics alpha cases the sign rows are judged against full-physics-alpha0 (judged_against) when that record exists, "
                    "and reported against ss-v4 in any case",
            "hypotheses": hypotheses,
            "key_quantities": list(KEY_QUANTITIES[case_id]),
            "reported_only": reported_only,
            "particle_band": PARTICLE_BAND,
            "absolute_band": ABSOLUTE_BAND,
            "per_cusp": {"planes_m": list(CUSP_PLANES_M), "half_width_m": CUSP_HALF_WIDTH_M,
                         "report": "per cusp plane (+-1 mm): electron and ion wall current, the axis-to-wall potential drop, the near-wall drop phi[wall - 3 cells] - phi[wall] "
                                   "(negative = virtual cathode), the near-wall T_e, the wall-ion mean impact energy; with SEE: effective yield, SEE current, mean emitted energy, "
                                   "SCL flag (effective yield >= 0.983 OR near-wall drop < 0); with the collision set: CEX / S over the window; with Coulomb: the NRL Spitzer "
                                   "nu_e at the window's peak cell and its ratio to nu_en, the electron-weighted Spitzer / pair-mean rates in the cusp columns; with the spatial "
                                   "gas: the local neutral density at the cusp plane (axis and channel mean), the metastable fraction and the stepwise share; and the "
                                   "ionisation centroid / quartiles with the fraction upstream of 12 mm - all beside the v4 values where they exist"},
            "effect_readings": {
                "coulomb": ["series coulomb block trailing-20 % means (nu_e_spitzer_peak_per_s, nu_e_spitzer_peak_over_nu_en, pair-mean nu_ee / nu_ei, nu_en, mean s, ln Lambda, pair counts)",
                            "maps coulomb_nu_ee_per_s / coulomb_nu_ei_per_s at the peak cell and the cusp columns; the Spitzer form recomputed from the n_e / T_e maps"],
                "neutrals": ["summary.neutral_inventory (spatial): channel-mean / anode / exit densities, atom ledger closure (neutral time), real-time plasma terms, gross / net utilisation, "
                             "neutral exit thrust, ceiling-violation fraction", "maps neutral_density_per_m3: axis profile, inner-third profile, anode / exit ratio, depletion vs the "
                             "initial Knudsen profile (channel mean)", "maps metastable_density_per_m3 / neutral_density_per_m3: axis fraction profile, max",
                             "summary.neutral_inventory.metastables: fraction of ground, stepwise fraction of S, production / superelastic / wall de-excitation rates"],
                "see": ["window_currents_a.see_emission_a / see_effective_yield; series see_* trailing means; cusps at or above the Hobbs-Wesson limit (count of 3)"],
                "collision_set": ["window_currents_a.cex_rate_per_s / mex_rate_per_s / fast-neutral rates; CEX / S; exit IEDF descriptors; level shares"],
                "anomalous": ["cumulative.anomalous event count and rate per electron (= alpha omega_ce at <|B|>)"],
                "ignition": ["summary.ignition (the gate's checks at 1.0 / 2.0 us: N_e and S ratios); the series N_e(t), I_d(t), S(t) at 0.1 / 0.5 / 1 / 2 us for the sustain reading"],
            },
        },
        "d_verdict": {
            "plateau_status": {
                "plateau_clean": "(a) AND (b): a quotable plateau of the case",
                "plateau_heating": "(a) but NOT (b): the plateau heats above 2 % (like the reference); the shifts are reported with the heating caveat",
                "no_plateau": "NOT (a) and not extinguished: budget / gate stop (the stop reason is classified: peak_debye_gate / residual_power / triad_drift / budget / other); "
                              "trailing-window quantities reported",
                "extinguished": "stop_reason no_ignition (the ignition gate), OR NOT (a) with the trailing-20 % N_e below 0.25 x the series maximum AND the trailing I_d below "
                                "0.25 x its running maximum (a late decay the latch never armed on): the discharge does not exist at this closure / operating point - a valid "
                                "recorded outcome, never relaunched (bitwise replay)",
            },
            "per_case_hypothesis_verdict": {
                "confirmed": "(a) reached AND every key quantity is CONFIRMING AND no hypothesis quantity with a band is CONTRADICTING; for the alpha cases: 'sustains' is the key "
                             "(the ignition gate passed AND (a) reached) and the sign rows against full-physics-alpha0 are secondary",
                "not_confirmed": "(a) reached AND at least one hypothesis quantity with a band is CONTRADICTING; OR for the alpha cases: extinguished (the sustain hypothesis is "
                                 "contradicted in the strongest form)",
                "inconclusive": "NOT (a) and not extinguished, OR (a) reached with no contradiction but some key quantity INSIDE THE BAND (the effect is smaller than the particle band)",
            },
            "note": "the verdict is about the SIGN of each effect on the recorded plateau and, for the alpha cases, about the EXISTENCE of the discharge; the magnitudes are the measurement. "
                    "No constant is 'tuned': the BN constants, the cross sections, the Coulomb operator and the gas model are the declared literature / spec values; alpha stays a declared closure parameter",
        },
        "e_sustain": {
            "applies_to": list(FULL_PHYSICS_CASES),
            "rule": "per alpha: sustains = ignition gate passed at 1.0 AND 2.0 us AND (a) reached; extinguished = plateau_status extinguished; undecided otherwise (budget / gate stop of a "
                    "live discharge). `assess --campaign` tabulates the three full-physics alphas beside the dilute-gas alpha-series outcomes (alpha-1over16 launch 1: extinguished, "
                    "record 0916a4f8; 1/64 and 0.345 as recorded when they finish) and the ext-val bohm-0.4 (alpha 0.345 at the static 2e20 gas) as the operating-point comparison: "
                    "'the Knudsen gas sustains the Bohm-leaky discharge at alpha = X' is stated per alpha as yes / no / undecided",
            "alpha_trend": "secondary: with >= 2 full-physics points at (a) the shifts vs full-physics-alpha0 are judged by the alpha-series signs (MONOTONE_QUANTITIES without n_g); "
                           "trend_confirmed needs all three points at (a) AND I_d and peak n_e monotone in the declared direction AND no monotone quantity CONTRADICTING; "
                           "trend_not_confirmed when three points reached and the order or a sign fails; inconclusive otherwise",
        },
        "f_qualification": {
            "applies_to": list(F_PAIR),
            "rule": "F = 10 is QUALIFIED when both members reached (a) AND for every plateau scalar (I_d, I_beam, S, gross utilisation, channel-mean n_g, peak n_e, T_e,peak) "
                    "|value(F10) - value(F1)| / |value(F1)| <= the particle band; DISQUALIFIED when both reached (a) and any scalar lies outside its band (the acceleration distorts "
                    "the plasma plateau: only F = 1 runs may then be quoted); not_evaluable unless both reached (a). The metastable fraction, the stepwise share and the depletion "
                    "fraction are EXPECTED to differ (the pool and the depletion run F x faster in plasma time) and are reported, not judged",
            "why": "neutral relaxation 0.2-2 ms >> 3 transits (7.2 us): a neutral steady state needs F ~ 100-300; F must first be shown not to move the plasma plateau while the gas "
                   "is quasi-static (see operating_point.neutrals.time_acceleration_note)",
        },
        "g_additivity": {
            "applies_to": "full-physics-alpha0 against its parts",
            "rule": "for every banded quantity: interaction = shift(full-physics-alpha0) - [shift(see-bn+xe-set-v2) + shift(coulomb) + shift_R5], with shift_R5 = shift(neutrals-spatial) "
                    "- shift(xe-set-v2) (the R5 case carries the set v2); the see-bn+xe-set-v2 and xe-set-v2 shifts are the physics-effects campaign's records (79a7c87a). ADDITIVE "
                    "when |interaction| <= band, SUPER_ADDITIVE when |interaction| > band with the sign of the sum of parts, SUB_ADDITIVE when it opposes the sum; the statement is "
                    "'additive' when every banded quantity is ADDITIVE, 'interacting' otherwise, 'not_evaluable' unless every part reached (a) and both physics-effects records exist. "
                    "R5 AS THE OPERATING-POINT CHANGE: beside the additivity, for every banded quantity whether |shift_R5| exceeds |shift(see-bn+xe-set-v2) + shift(coulomb)| "
                    "('the operating point dominates') is reported - the audit's expectation is that it does on S, utilisation and peak n_e",
            "hypothesis": "no interaction sign is predeclared for SEE x Coulomb x set; the R5 x SEE interaction is expected SUPER-additive on I_d (denser plasma at the SEE-lowered "
                          "sheaths) - reported as a measurement, the statement is not part of any verdict",
        },
    }
    dropped = ("development/screening run", "single seed and a single refined grid", "preregistered resolution-convergence")
    if case["collision_set"]:
        dropped += ("no ion-neutral collisions", "electron-neutral: isotropic elastic")
    if case["see"]:
        dropped += ("dielectric wall: perfectly absorbing",)
    if case["neutrals"]:
        dropped += ("neutrals: 0-D quasi-steady inventory",)
    simplifications = [s for s in p["simplifications"] if not s.startswith(dropped)]
    if alpha > 0.0:
        simplifications = [s.replace("electrostatic only, azimuthally symmetric: no azimuthal instabilities/anomalous transport",
                                     "electrostatic only, azimuthally symmetric: no azimuthal instabilities; the anomalous cross-field transport they would drive is IMPOSED as a declared "
                                     "Bohm-type closure (alpha), not computed") for s in simplifications]
    if case["collision_set"]:
        simplifications += [
            ("electron-neutral: isotropic elastic without the 2 m_e/M energy loss, FOUR Biagi-v7.1 excitation levels (8.315 / 9.447 / 9.917 / 11.7 eV), single ionisation with "
             "Vahedi-Surendra secondary energy and isotropic emission (model v2.3.0)" + ("; the Xe(6s) metastable pool with BEB stepwise ionisation and superelastic return (model v2.5.0)"
             if case["neutrals"] else "; no metastable pool, no stepwise ionisation")),
            ("ion-neutral: Xe+ - Xe charge exchange (Miller 2002) and momentum transfer (Phelps isotropic) against the " + ("LOCAL published gas density with the local gas drift / "
             "thermal speed; CEX fast neutrals become neutral particles at F x W" if case["neutrals"] else "0-D inventory density with a Maxwellian atom at the gas temperature; CEX fast "
             "neutrals are booked by a straight-line march, not tracked as particles") + "; no Xe2+"),
        ]
    if case["see"]:
        simplifications += [("dielectric wall: secondary electron emission from BN (Vaughan fit of Villemant 2019 + Sydorenko split, T_see 2 eV, ion-induced yield 0, no Hobbs-Wesson cap - the "
                             "space-charge-limited state emerges) with accumulated surface charge = absorbed - emitted; zero field in the dielectric backing; no sputtering, no conditioning")]
    if case["coulomb"]:
        simplifications += ["Coulomb collisions: e-e and e-i binary pairing every 10 steps (Takizuka-Abe / Nanbu) with the NRL logarithm from cell moments; i-i off; no e-i energy-exchange "
                            "beyond the exact kinematics (the ions are cold: the 1e-9 K_e ledger remainder is recorded)"]
    if case["neutrals"]:
        simplifications += [
            ("neutrals: free-molecular test particles (Kn ~ 10-100; no neutral-neutral collisions), diffuse reflection at T_w 500 K with full accommodation, cosine-law feed at 300 K from "
             "the anode face, wall-ion recycling at the impact cell, nearest-cell density deposit (~13 % shot noise per cell at ~60 macro-neutrals) clamped at the MCC ceiling"),
            (f"time acceleration F = {case['time_acceleration']:g}: " + ("PHYSICAL neutral time - the gas is quasi-frozen at the initial Knudsen profile over 3 transits (depletion < 1 %); "
             "the recorded plateau is the plasma response to that profile, NOT a neutral steady state" if case["time_acceleration"] == 1.0 else
             "the qualification member; the gas responds 10 x faster than physical (depletion ~3 %, the metastable pool fills in ~1 us of plasma time)")),
        ]
    if alpha > 0.0:
        simplifications += ["the Bohm closure is a constant alpha everywhere (the probability follows the local |B| only); a per-cell effective mobility from companion instability-plane "
                            "runs is the declared follow-up (audit section 6 ii)"]
    else:
        simplifications += ["no anomalous cross-field transport (alpha = 0): every discharge quantity here is conditional on alpha = 0 (audit section 6); the full-physics alpha cases carry the closure"]
    simplifications += [
        "single seed per case: the shifts are judged against the recorded 50 um particle band, not a per-case replicate; the IEDF fraction and the ionisation centroid have declared absolute bands",
        "the reference (ss-v4) heats at +2.46 % of the electrode work on the corrected ledger: differences against it carry that caveat",
        "preregistered physics-effect / operating-point / closure study of a development model: no experimental validation, not a performance prediction",
    ]
    p["simplifications"] = simplifications
    p["claim_boundary"] = ("preregistered full-physics campaign (Coulomb collisions alone; the spatial Knudsen-profile gas with metastables at F = 1 and its F = 10 qualification "
                           "twin; every R2-R5 effect together at alpha = 0, 1/16 and 0.345 with the Bohm perpendicular-rotation closure) on the reference design at 33.3 um / 1.4 ps / "
                           "W 2.667e4 and the ss-v4 feed / injection, against the recorded ss-v4 plateau; the outcomes are (i) whether the full-physics discharge EXISTS at each "
                           "alpha at this operating point (ignition gate + plateau), (ii) the SIGN of each effect on I_d, S, utilisation, n_g (profile), I_beam, peak n_e, T_e,peak, the "
                           "exit IEDF, the ionisation centroid, the per-cusp sheath / SEE / Coulomb / metastable readings, with the magnitudes recorded, (iii) whether F = 10 moves the "
                           "plateau, (iv) whether the effects add; every discharge quantity of the 2D axisymmetric model is conditional on the declared alpha; the F = 1 gas is quasi-"
                           "frozen (no neutral steady state is claimed); not validated against experiment; not a thruster performance prediction")
    p["reference_run"] = v4_reference_block()
    p["preregistration"] = {
        "protocol": "protocols/<case>.json is frozen at the preregistration commit (its sha256 is listed in the campaign protocol.json); summary.json records protocol_sha256 and "
                    "git_head; run.py launch refuses a dirty worktree, a HEAD that is not the recorded preregistration commit, a sealed protocol that differs from its recomposition, "
                    "or an existing execution lock",
        "preflight": "preflight-<case>.json on the launch box (real P2 field on the 90 x 720 grid, mesh, factorisation, memory incl. the neutral particles, ms/step at the seed load, at a "
                     "synthetic ~4.5 M-particle load and - spatial cases - at ~7.7 M, GPU load before, MPS clients) - non-evidentiary; the budget is derived from it",
        "shakedown": "shakedown-<case>.json: a 100 000-step real-input run of EVERY case (results-shakedown/<case>/, cadences shrunk, every gate live) through run -> finalize -> "
                     "assess (case and campaign) on the launch box - non-evidentiary, not committed beyond the record. The R4 / R5 shakedowns (82255081 / 55092f4c) ran the R3 "
                     "protocol with F = 100 and other trees: NOT reused",
        "one_execution": "one detached launch per case from the scheduler's worktree at the preregistration commit, one MPS slot each, in the declared priority order as slots free, "
                         "chained AFTER the physics-effects queue (pe-queue) by the box slot-waiter (fp-queue); a wall-budget stop may be resumed (new session, same identity, "
                         "disclosed); no parameter is changed after the freeze",
    }
    return p


def protocol_sha256(protocol: dict[str, Any]) -> str:
    """sha256 of the sealed file bytes (``canonical_bytes + newline`` is exactly what ``write_sealed_protocols`` writes)."""

    return sha256(canonical_bytes(protocol) + b"\n").hexdigest()


def compose_campaign(case_protocols: dict[str, dict[str, Any]], *, budgets: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The campaign-level protocol: design, sealed case hashes, launch order, acceptance (mirrors the per-case blocks), amendments."""

    first = case_protocols[LAUNCH_PRIORITY[0]] if LAUNCH_PRIORITY[0] in case_protocols else next(iter(case_protocols.values()))
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "preregistered_full_physics_r4_r5_combined_not_validated",
        "title": "Full physics v1: Coulomb (R4), the spatial Knudsen gas with metastables (R5) and its time-acceleration qualification, and every R2-R5 effect together at alpha = 0, 1/16, 0.345 "
                 "on the 33 um reference plateau",
        "model_version": MODEL_VERSION,
        "model_spec": MODEL_SPEC,
        "design": {
            "template": "modern/experiments/pic2d_cft_steady_state_v4/protocol.json (reference design divergent-exit-stack, 90 x 720 at 33.33 um, dt 1.4 ps, W 26 666.7, seed 20260903, frames ON)",
            "cases": {k: {kk: v[kk] for kk in ("label", "effects", "see", "collision_set", "coulomb", "neutrals", "time_acceleration", "alpha", "group", "role")} for k, v in CASES.items()},
            "reference_point": "the recorded ss-v4 plateau (0d228ad2): SEE off, legacy collision set, 0-D gas, no Coulomb, alpha = 0, K = 1 - NOT re-run",
            "launch_priority": list(LAUNCH_PRIORITY),
            "launch_priority_note": "the sustain question first (full-physics-alpha0.345: the strongest leak at the Knudsen gas), then the full model's alpha = 0 point, the R5 operating-point "
                                    "case, the classical-Bohm full-physics point, the Coulomb-alone case, and the F = 10 qualification twin last; each takes one H100 MPS slot AFTER the "
                                    "physics-effects queue (pe-queue: see-bn -> xe-set-v2 -> see-bn+xe-set-v2) as the scheduler frees one; never a 5th client",
            "coulomb_block": COULOMB_BLOCK,
            "see_block": SEE_BN_BLOCK,
            "collision_set_block": COLLISION_SET_V2_BLOCK,
            "neutrals_block_f1": {k: v for k, v in neutrals_block(1.0).items() if not k.endswith("_note")},
            "mcc_ceiling_spatial_per_m3": MCC_CEILING_SPATIAL_PER_M3,
            "knudsen_anode_density_per_m3": KNUDSEN_ANODE_DENSITY_PER_M3,
            "alphas": {k: v["alpha"] for k, v in CASES.items()},
            "hypotheses_by_case": HYPOTHESES_BY_CASE,
            "key_quantities_by_case": KEY_QUANTITIES,
            "cusp_planes_m": list(CUSP_PLANES_M),
        },
        "acceptance": {case: p["stopping_rule"]["acceptance"] for case, p in case_protocols.items()},
        "reference_run": first["reference_run"],
        "sealed_protocols": {f"modern/experiments/pic2d_full_physics_v1/protocols/{case}.json": protocol_sha256(p) for case, p in case_protocols.items()},
        "budgets": budgets or {case: {"wall_budget_seconds": p["stopping_rule"]["wall_budget_seconds"], "note": p["stopping_rule"]["wall_budget_note"]} for case, p in case_protocols.items()},
        "identity_policy": {
            "reference": "no effect block, K = 1 -> config_sha256 f10772b25b03... = the ss-v4 record (test-pinned): the campaign's reference IS the recorded run",
            "coulomb": "numerics.coulomb enters config_sha256 through CoulombConfig.to_dict",
            "see": "numerics.see enters config_sha256 through SEEConfig.to_dict",
            "collision_set": "operating_point.collision_set enters config_sha256 through MCCConfig.to_dict()['collision_set'] (recomputed payload hashes)",
            "neutrals": "operating_point.neutrals enters config_sha256 through SpatialNeutralConfig.to_dict (incl. time_acceleration: F = 1 and F = 10 are different identities) and the "
                        "raised MCC ceiling through MCCConfig; the 0-D inventory block is absent",
            "alpha": "numerics.anomalous_collisions {model, alpha} enters config_sha256 for the alpha > 0 cases",
            "k_5": "moment_sample_interval 5 enters config_sha256 (v2.0.5 policy)",
            "debye_floor": "min_accumulated_macro_particle_steps_at_peak 64000 enters config_sha256 only because it is declared (v2.0.6 policy)",
            "arming_and_ignition": "stopping_rule keys (drift_members_arming, ignition_gate) are OUTSIDE config_sha256: they change when a run stops, never what it computes",
            "ledger": "the v2.0.6 W correction is code (bug fix, identity unchanged); acceptance (b) reads the corrected statistic natively",
        },
        "preregistration": first["preregistration"],
        "amendments": [],
    }


def write_sealed_protocols(case_protocols: dict[str, dict[str, Any]], campaign: dict[str, Any]) -> list[Path]:
    PROTOCOLS_DIR.mkdir(exist_ok=True)
    written = []
    for case, p in case_protocols.items():
        path = PROTOCOLS_DIR / f"{case}.json"
        path.write_bytes(canonical_bytes(p) + b"\n")
        written.append(path)
    CAMPAIGN_PROTOCOL_PATH.write_bytes(canonical_bytes(campaign) + b"\n")
    written.append(CAMPAIGN_PROTOCOL_PATH)
    return written


def load_case_protocol(case_id: str) -> dict[str, Any]:
    path = PROTOCOLS_DIR / f"{case_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"sealed protocol {path} missing; run `run.py compose` first")
    return json.loads(path.read_text(encoding="utf-8"))


def load_campaign() -> dict[str, Any]:
    return json.loads(CAMPAIGN_PROTOCOL_PATH.read_text(encoding="utf-8"))


__all__ = [
    "ABSOLUTE_BAND", "AT_RESULTS", "CASES", "CENTROID_BAND_M", "COULOMB_BLOCK", "CUSP_HALF_WIDTH_M", "CUSP_PLANES_M", "DRIFT_MEMBERS_ARMING", "EXPERIMENT_ID",
    "FULL_PHYSICS_CASES", "F_PAIR", "F_QUALIFICATION_FACTOR", "HYPOTHESES_BY_CASE", "HYPOTHESES_COULOMB", "HYPOTHESES_FULL_ALPHA", "HYPOTHESES_FULL_ALPHA0", "HYPOTHESES_R5",
    "IEDF_FRACTION_BAND", "IGNITION_GATE", "KEY_QUANTITIES", "KNUDSEN_ANODE_DENSITY_PER_M3", "KNUDSEN_CHANNEL_MEAN_PER_M3", "LAUNCH_PRIORITY", "MCC_CEILING_SPATIAL_PER_M3",
    "MONOTONE_QUANTITIES", "NEUTRAL_MACRO_WEIGHT", "PARTICLE_BAND", "PE_RESULTS", "PLATEAU_SCALARS", "QUANTITY_KEYS", "REFERENCE_CASE", "REFERENCE_CORRECTED_RESIDUAL",
    "REPORTED_ONLY_SPATIAL", "SPATIAL_CASES", "STEPS_TO_3_TRANSITS", "V4_RESULTS", "channel_wall_cells", "compose_campaign", "compose_case_protocol", "iedf_low_energy_fraction",
    "ionization_centroid_from_maps", "load_campaign", "load_case_protocol", "load_v4_protocol", "neutrals_block", "protocol_sha256", "reference_extras_from_v4",
    "v4_reference_block", "wall_area_m2", "wall_power_and_ion_energy", "write_sealed_protocols",
]
