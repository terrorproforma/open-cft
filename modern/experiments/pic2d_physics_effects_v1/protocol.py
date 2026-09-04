"""Composition of the physics-effects run protocols on the steady-state v4 (33 um) template (roadmap R2 + R3).

Each case is the ss-v4 protocol (``pic2d_cft_steady_state_v4/protocol.json``: reference design, 90 x 720 cells at 33.33 um,
dt 1.4 ps, W 26 666.7, the v1.3 closure, seed 20260903, frames ON) with exactly these changes:

* ``see-bn``: ``numerics.see`` = model v2.2.0 ``see_dielectric_v1`` with the BN preset (Vaughan fit of Villemant 2019, Sydorenko
  component split, T_see 2 eV, ion-induced yield 0, no Hobbs-Wesson cap); the legacy lumped collision set stays;
* ``xe-set-v2``: ``operating_point.collision_set`` = model v2.3.0 ``xe_collision_set_v2`` (four Biagi-v7.1 excitation levels + Xe+ / Xe
  CEX and MEX with the fast-neutral contract); the wall stays absorbing (SEE off);
* ``see-bn+xe-set-v2``: both blocks;
* every case: the v2.0.6 gates (the peak-Debye gate's accumulated-particle-step floor 64 000 declared; the corrected energy ledger is
  code, so acceptance (b) reads the CORRECTED residual natively), ``numerics.performance.moment_sample_interval`` K = 5 (v2.0.5), the
  acceptance block (plateau rule, corrected residual < +2 %, the shift table vs the recorded ss-v4 plateau with the particle band, the
  per-cusp report with the SEE / collision diagnostics, the audit's signs as hypotheses, the per-case and combined verdict rules), the
  wall budget from the launch-box measured rate x 1.5 (filled from the preflight before the preregistration commit).

NO anomalous-transport block: alpha = 0, so each effect is isolated against the recorded ss-v4 plateau (0d228ad2), which fails its own
acceptance (b) at +2.46 % on the corrected ledger - stated in every sealed protocol.  Everything else (geometry, operating point, dt,
grid, W, seed, cadences, plateau rule, the v2.0.3 gate thresholds) is byte-for-byte the v4 protocol; ``test_pic2d_physics_effects_v1.py``
pins that.
"""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from cft_revival.orbit_mc.artifacts import canonical_bytes
from cft_revival.pic2d.cross_sections_xe import (
    XE_COLLISION_SET_V2_NAME,
    XE_ELECTRON_SET_V2_FILE,
    XE_ELECTRON_SET_V2_PAYLOAD_SHA256,
    XE_ION_NEUTRAL_SET_V1_FILE,
    XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256,
)
from cft_revival.pic2d.models import ELEMENTARY_CHARGE_C as _E_C
from cft_revival.pic2d.see import (
    DEFAULT_MAX_EMITTED_PER_IMPACT,
    HOBBS_WESSON_CRITICAL_YIELD_XE,
    MATERIALS,
    SEEConfig,
)

HERE = Path(__file__).resolve().parent
MODERN = HERE.parents[1]
V4_DIR = MODERN / "experiments" / "pic2d_cft_steady_state_v4"
V4_PROTOCOL_PATH = V4_DIR / "protocol.json"
V4_RESULTS = V4_DIR / "results"
V5_PROTOCOL_PATH = MODERN / "experiments" / "pic2d_cft_steady_state_v5" / "protocol.json"
PROTOCOLS_DIR = HERE / "protocols"
CAMPAIGN_PROTOCOL_PATH = HERE / "protocol.json"

SCHEMA_VERSION = "cft-revival.pic2d-physics-effects-v1.protocol/0.1.0"
CASE_SCHEMA_VERSION = "cft-revival.pic2d-physics-effects-v1.case-protocol/0.1.0"
EXPERIMENT_ID = "pic2d-physics-effects-v1"
MODEL_VERSION = ("pic2d v1.3 closure (quasi-steady 0-D neutral inventory, NO wall-ion recycling) + model v2.2.0 see_dielectric_v1 (secondary electron "
                 "emission from the BN dielectric wall; per case) + model v2.3.0 xe_collision_set_v2 (four excitation levels + Xe+ / Xe CEX and MEX with the "
                 "fast-neutral contract; per case) with the v2.0.6 gates (window-mode peak-Debye gate with the accumulated-particle-step floor, windowed "
                 "residual-power gate on the W-corrected ledger) and the v2.0.5 K = 5 moment sampling; NO anomalous-transport closure (alpha = 0)")
MODEL_SPEC = ("modern/spec/pic2d/pic2d-model-v1.3.json (physics) + modern/spec/pic2d/pic2d-model-v2.0.json (gates_v2_0: v2.0.3 gates, v2.0.6 ledger "
              "correction + Debye floor, v2.0.5 performance block) + modern/spec/pic2d/pic2d-model-v2.2.json (see_dielectric_v1) + "
              "modern/spec/pic2d/pic2d-model-v2.3.json (xe_collision_set_v2)")

# the P2 divergent-exit-stack cusp planes (cusp topology search v3.1, cec47f12; the ss-v4 dashboard's overlay) for the per-cusp report
CUSP_PLANES_M: tuple[float, ...] = (6.028e-3, 12.000e-3, 17.972e-3)
CUSP_HALF_WIDTH_M = 1.0e-3
REFERENCE_CASE = "ss-v4"
REFERENCE_CORRECTED_RESIDUAL = 0.02458578453535502      # ss-v4 ledger-corrected.json end_state_window.corrected_ratio (+2.46 %)
STEPS_TO_3_TRANSITS = 5_142_858
IEDF_LOW_ENERGY_FRACTION_OF_ANODE = 0.1                  # "low-energy" exit ions: bin centre below 0.1 x the anode potential (30 eV at 300 V)
IEDF_FRACTION_BAND = 0.03                                # DECLARED absolute detection band for the low-energy fraction (no replicate exists)

# the two physics blocks exactly as the runner reads them (numerics.see -> SEEConfig; operating_point.collision_set -> CollisionSetConfig)
SEE_BN_BLOCK: dict[str, Any] = {
    "enabled": True, "material": "BN", "yield_model": "vaughan_components", "constant_yield": 0.0, "constant_yield_threshold_ev": 0.0,
    "emission_temperature_ev": 2.0, "ion_induced_yield": 0.0, "max_emitted_per_impact": DEFAULT_MAX_EMITTED_PER_IMPACT,
    "space_charge_limit_yield": HOBBS_WESSON_CRITICAL_YIELD_XE, "overrides": {},
}
COLLISION_SET_V2_BLOCK: dict[str, Any] = {"name": XE_COLLISION_SET_V2_NAME, "ion_neutral": True}

# the three sealed cases (roadmap R2 = SEE, R3 = collision set; the third isolates the interaction)
CASES: dict[str, dict[str, Any]] = {
    "see-bn": {"see": True, "collision_set": False, "label": "SEE(BN)", "effects": ["see_dielectric_v1"],
               "role": "R2 alone: secondary electron emission from the BN dielectric wall (model v2.2.0) on the legacy lumped collision set - the audit's "
                       "gap (a) isolated against the recorded ss-v4 plateau"},
    "xe-set-v2": {"see": False, "collision_set": True, "label": "Xe set v2", "effects": ["xe_collision_set_v2"],
                  "role": "R3 alone: four Biagi-v7.1 excitation levels + Xe+ / Xe charge exchange and momentum transfer with the fast-neutral contract "
                          "(model v2.3.0) on the absorbing wall - the audit's gaps (e1) + (e4) isolated against the recorded ss-v4 plateau"},
    "see-bn+xe-set-v2": {"see": True, "collision_set": True, "label": "SEE(BN) + Xe set v2", "effects": ["see_dielectric_v1", "xe_collision_set_v2"],
                         "role": "both effects together: the combined shift is compared with the SUM of the two single-effect shifts (additivity statement) - "
                                 "the closure the later campaigns (R4 Coulomb, R5 neutrals) will build on"},
}
LAUNCH_PRIORITY: tuple[str, ...] = ("see-bn", "xe-set-v2", "see-bn+xe-set-v2")

# the quantities of the shift table: relative shifts with the ss-v4 particle band where one is measured, absolute for the IEDF fraction
QUANTITY_KEYS: tuple[str, ...] = ("discharge_current_a", "exit_ion_beam_a", "ionization_rate_per_s", "gross_utilisation", "neutral_density_per_m3",
                                  "peak_n_e_window_per_m3", "t_e_peak_window_ev", "iedf_low_energy_fraction", "anode_ion_a",
                                  "wall_electron_power_w", "wall_ion_mean_energy_ev")
# the ss-v4 particle band (seed-b / W x 0.7 of the 50 um pair; the v4 c-tolerances are 2x): a shift inside the band is not an effect
PARTICLE_BAND: dict[str, float] = {"discharge_current_a": 0.057, "exit_ion_beam_a": 0.057, "ionization_rate_per_s": 0.046, "gross_utilisation": 0.046,
                                   "neutral_density_per_m3": 0.040, "peak_n_e_window_per_m3": 0.119, "t_e_peak_window_ev": 0.093}
ABSOLUTE_BAND: dict[str, float] = {"iedf_low_energy_fraction": IEDF_FRACTION_BAND}

# predeclared expectations per effect (physics audit 0901138a sections 4.a / 4.e, spec v2.2 predeclared_hypotheses_from_the_audit, spec v2.3
# predeclared_expectations): the SIGN is the hypothesis ("+" up, "-" down, "0" unchanged inside the band), the magnitude is what the run measures
HYPOTHESES_SEE: dict[str, dict[str, Any]] = {
    "discharge_current_a": {"sign": "+", "expected": "+10 to +30 %", "reason": "more electrons reach the anode through the lowered cusp sheaths; near-wall conductivity"},
    "t_e_peak_window_ev": {"sign": "-", "expected": "-10 to -25 %", "reason": "the emitted cold secondaries (2 eV) and the lower sheath drops cool the hot tail"},
    "peak_n_e_window_per_m3": {"sign": "-", "expected": "-5 to -15 %", "reason": "the wall no longer confines the electrons: the density that builds before the cusps falls"},
    "cusp_sheath_drop_v": {"sign": "-", "expected": "-10 to -45 % (delta_eff 0.3-0.9; Hobbs-Wesson factor T_e ln(1/(1 - delta_eff)))",
                           "reason": "the emitted electron current lowers the floating-wall potential drop; per cusp, read beside the effective yield and the SCL flag"},
    "wall_electron_power_w": {"sign": "+", "expected": "x1.5-2", "reason": "the lower sheath admits more primaries; the emitted flux returns part of the energy at T_see"},
    "wall_ion_mean_energy_ev": {"sign": "-", "expected": "down by the sheath fraction", "reason": "ions fall through a smaller sheath drop"},
}
HYPOTHESES_XE: dict[str, dict[str, Any]] = {
    "discharge_current_a": {"sign": "0", "expected": "unchanged (< 5 %, inside the 5.7 % band)", "reason": "CEX / MEX act on the ions; the four levels change the "
                            "electron energy loss per event only (~15 % more inelastic power)"},
    "ionization_rate_per_s": {"sign": "-", "expected": "-3 to -5 % (inside the 4.6 % band): the per-event loss 8.32 -> 9.4-10.1 eV",
                              "reason": "R3a: the same total excitation frequency removes more energy per event"},
    "gross_utilisation": {"sign": "-", "expected": "-3 to -5 % (follows S)", "reason": "utilisation = S / feed"},
    "t_e_peak_window_ev": {"sign": "-", "expected": "-3 to -5 % (inside the 9.3 % band)", "reason": "R3a: higher inelastic power per excitation"},
    "iedf_low_energy_fraction": {"sign": "+", "expected": "+0.15 to +0.30 absolute", "reason": "lambda_CEX ~ 62 mm at 3e19 vs 12-24 mm flight paths: "
                                 "1 - exp(-L / lambda) = 0.18-0.32 of the channel-born ions exchange and reach the exit plane slow"},
    "anode_ion_a": {"sign": "+", "expected": "up (slow CEX ions born near the anode)", "reason": "R3b: slow CEX ions upstream of the cusps fall to the anode"},
    "fast_neutral_exit_rate_per_s": {"sign": "+", "expected": "> 0 (reference 0 by construction); F / S ~ the exchanged fraction x the exit fraction",
                                     "reason": "CEX fast neutrals leaving through the exit aperture carry the exchanged ion momentum (thrust redistributed)"},
}
HYPOTHESES_COMBINED: dict[str, dict[str, Any]] = {
    **{k: v for k, v in HYPOTHESES_XE.items() if k not in ("discharge_current_a", "t_e_peak_window_ev")},
    **HYPOTHESES_SEE,
    "discharge_current_a": {"sign": "+", "expected": "+10 to +30 % (SEE dominates; the collision set leaves I_d inside the band)",
                            "reason": HYPOTHESES_SEE["discharge_current_a"]["reason"]},
    "t_e_peak_window_ev": {"sign": "-", "expected": "-10 to -30 % (both effects cool)", "reason": "SEE (-10 to -25 %) and the per-event excitation loss (-3 to -5 %)"},
}
# the quantities whose CONFIRMING status the per-case verdict requires (every other hypothesis with a band can only CONTRADICT)
KEY_QUANTITIES: dict[str, tuple[str, ...]] = {
    "see-bn": ("discharge_current_a", "t_e_peak_window_ev"),
    "xe-set-v2": ("iedf_low_energy_fraction", "discharge_current_a"),
    "see-bn+xe-set-v2": ("discharge_current_a", "t_e_peak_window_ev", "iedf_low_energy_fraction"),
}
HYPOTHESES_BY_CASE: dict[str, dict[str, dict[str, Any]]] = {"see-bn": HYPOTHESES_SEE, "xe-set-v2": HYPOTHESES_XE, "see-bn+xe-set-v2": HYPOTHESES_COMBINED}


def load_v4_protocol() -> dict[str, Any]:
    return json.loads(V4_PROTOCOL_PATH.read_text(encoding="utf-8"))


def iedf_low_energy_fraction(counts: np.ndarray, edges_ev: np.ndarray, anode_v: float, fraction_of_anode: float = IEDF_LOW_ENERGY_FRACTION_OF_ANODE) -> float | None:
    """Fraction of the exit-plane macro-ion counts whose bin centre lies below ``fraction_of_anode x anode_v`` (None without counts)."""

    counts = np.asarray(counts, dtype=np.float64)
    edges = np.asarray(edges_ev, dtype=np.float64)
    total = float(counts.sum())
    if counts.size == 0 or total <= 0.0:
        return None
    centres = 0.5 * (edges[:-1] + edges[1:])
    return float(counts[centres < fraction_of_anode * anode_v].sum() / total)


def wall_area_m2(grid, z_cells: np.ndarray) -> np.ndarray:
    """Wall area of each axial wall cell (bore cylinder or cone slant) - the v4 dashboards' convention."""

    geometry = grid.geometry
    radius = np.array([float(geometry.wall_radius_m(z)) if z < geometry.z_max_m else float(geometry.exit_radius_m) for z in z_cells])
    if geometry.cone_start_z_m < geometry.z_max_m:
        slope = (geometry.exit_radius_m - geometry.bore_radius_m) / (geometry.z_max_m - geometry.cone_start_z_m)
        slant = np.where(z_cells > geometry.cone_start_z_m, np.sqrt(1.0 + slope * slope), 1.0)
    else:
        slant = np.ones_like(z_cells)
    return 2.0 * np.pi * radius * grid.dz_m * slant


def channel_wall_cells(maps: dict[str, np.ndarray], grid) -> tuple[int, np.ndarray, np.ndarray]:
    """(number of channel wall cells, their axial centres, their areas) for the wall profiles of ``maps``."""

    geometry = grid.geometry
    n_wall = min(int(np.asarray(maps["wall_electron_flux_per_m2_s"]).size), round(geometry.channel_length_m / grid.dz_m))
    z_cells = geometry.z_min_m + (np.arange(n_wall) + 0.5) * grid.dz_m
    return n_wall, z_cells, wall_area_m2(grid, z_cells)


def wall_power_and_ion_energy(maps: dict[str, np.ndarray], grid) -> tuple[float | None, float | None]:
    """Window-averaged electron power onto the channel wall (W) and the wall-ion-flux-weighted mean ion impact energy (eV) from the wall profiles."""

    if not all(k in maps for k in ("wall_electron_flux_per_m2_s", "wall_electron_mean_energy_ev", "wall_ion_flux_per_m2_s", "wall_ion_mean_energy_ev")):
        return None, None
    n_wall, _, area = channel_wall_cells(maps, grid)
    flux_e = np.nan_to_num(np.asarray(maps["wall_electron_flux_per_m2_s"], dtype=float)[:n_wall])
    energy_e = np.nan_to_num(np.asarray(maps["wall_electron_mean_energy_ev"], dtype=float)[:n_wall])
    flux_i = np.nan_to_num(np.asarray(maps["wall_ion_flux_per_m2_s"], dtype=float)[:n_wall])
    energy_i = np.nan_to_num(np.asarray(maps["wall_ion_mean_energy_ev"], dtype=float)[:n_wall])
    power_w = float(np.sum(_E_C * flux_e * area * energy_e))
    weight = float(np.sum(flux_i * area))
    ion_energy = float(np.sum(flux_i * area * energy_i) / weight) if weight > 0.0 else None
    return power_w, ion_energy


def reference_extras_from_v4(results: Path = V4_RESULTS, anode_v: float = 300.0) -> dict[str, Any] | None:
    """The ss-v4 reference values this campaign adds to the v5 pinned block (IEDF low-energy fraction, anode ion current, wall electron power,
    flux-weighted wall ion impact energy), recomputed from the v4 artifacts on disk (None when they are not checked out)."""

    if not (results / "maps.npz").is_file() or not (results / "summary.json").is_file():
        return None
    from experiments.pic2d_cft_steady_state_v1 import (
        run as runner,  # local: the runner is heavy and imports nothing from here
    )

    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    with np.load(results / "maps.npz") as archive:
        maps = {k: np.asarray(archive[k]) for k in archive.files}
    grid = runner.build_config(load_v4_protocol(), backend="cpu").grid
    power, ion_energy = wall_power_and_ion_energy(maps, grid)
    return {
        "iedf_low_energy_fraction": iedf_low_energy_fraction(maps["iedf_ion_counts"], maps["iedf_edges_ev"], anode_v),
        "anode_ion_a": float(summary["window_currents_a"]["anode_ion_a"]),
        "wall_electron_power_w": power,
        "wall_ion_mean_energy_ev": ion_energy,
    }


def v4_reference_block() -> dict[str, Any]:
    """The reference point of every case: the recorded ss-v4 plateau (0d228ad2; SEE off, legacy collision set, alpha = 0) with its CORRECTED ledger status."""

    v5 = json.loads(V5_PROTOCOL_PATH.read_text(encoding="utf-8"))["reference_run"]
    quantities = dict(v5["quantities"])
    extras = reference_extras_from_v4()
    if extras is not None:
        quantities.update(extras)
    return {
        "case": REFERENCE_CASE, "see": None, "collision_set": "legacy (xenon-cross-sections-v1.json, collisionless ions)", "anomalous_collisions": None,
        "experiment": "modern/experiments/pic2d_cft_steady_state_v4", "results_dir": "modern/experiments/pic2d_cft_steady_state_v4/results",
        "commit": v5["commit"], "run_git_head": v5["run_git_head"], "protocol_sha256_prefix": v5["protocol_sha256_prefix"], "config_sha256_prefix": v5["config_sha256_prefix"],
        "grid": v5["grid"], "plateau": v5["plateau"], "quantities": quantities,
        "quantities_added_by_this_campaign": ["iedf_low_energy_fraction (exit-plane IEDF counts with bin centre < 0.1 x anode = 30 eV, from maps.npz)",
                                              "anode_ion_a (summary.window_currents_a)", ("wall_electron_power_w (e x wall_electron_flux x area x wall_electron_mean_energy, "
                                              "summed over the channel wall cells of maps.npz)"), "wall_ion_mean_energy_ev (wall-ion-flux-weighted mean of wall_ion_mean_energy_ev)"],
        "corrected_ledger": {
            "sidecar": "modern/experiments/pic2d_cft_steady_state_v4/results/ledger-corrected.json (v2.0.6, 02013df0)",
            "windowed_residual_over_electrode_work_corrected": REFERENCE_CORRECTED_RESIDUAL,
            "acceptance_b_below_0p02": False,
            "statement": "the reference FAILS its own acceptance (b) on the corrected ledger: +2.46 % of the electrode work in the trailing 400 000-step window "
                         "at the stop (recorded -7.67 % before the v2.0.6 W fix; still rising at the plateau). It is the campaign's SEE-off / legacy-set point "
                         "AS RECORDED, not a clean reference: every difference reported against it carries this caveat, and a case that passes (b) while the "
                         "reference does not is itself a finding about the effect (less finite-grid heating at a lower peak density or temperature)",
        },
        "particle_band": PARTICLE_BAND,
        "absolute_band": ABSOLUTE_BAND,
        "particle_band_note": "the 50 um convergence pair's relative bands (seed-b / W x 0.7; the v4 c-tolerances are 2x these): a shift smaller than the band is "
                              "reported as 'inside the particle band' and does not count as an effect; the IEDF low-energy fraction has NO replicate band - the "
                              f"declared absolute detection band {IEDF_FRACTION_BAND} (~40 % of the reference fraction 0.067, > 3x the largest relative particle "
                              "band applied to it) is a preregistered choice; anode_ion_a, wall_electron_power_w and wall_ion_mean_energy_ev have no band: reported "
                              "with their hypothesis signs, never judged",
    }


def _see_block() -> dict[str, Any]:
    block = copy.deepcopy(SEE_BN_BLOCK)
    material = MATERIALS["BN"]
    block["see_note"] = (
        "model v2.2.0 see_dielectric_v1 (spec pic2d-model-v2.2.json): electron impacts on the dielectric wall (boundary code 3: channel wall + cone stair) emit "
        "floor(delta) + Bernoulli(frac) macro-electrons of the impacting weight from the Vaughan yield delta(E, theta) with the BN constants delta_max "
        f"{material.delta_max} at {material.energy_max_ev:.0f} eV, threshold {material.energy_threshold_ev} eV, k_rise {material.k_rise} (Vaughan fit of Villemant et al. "
        "2019 as tabulated by PICLas; first crossover 35.7 eV vs Dunaevsky 2003's 35 eV; flux-averaged yield 0.48 / 0.58 / 0.69 at T_e 5 / 7 / 10 eV, critical "
        f"temperature 20.3 eV), Sydorenko 2006 split (elastic {material.elastic_fraction}, inelastic {material.inelastic_fraction}, the rest true secondaries as a "
        "flux half-Maxwellian at T_see 2 eV, cosine law about the inward normal), ion-induced yield 0, NO Hobbs-Wesson cap (the space-charge-limited state emerges; "
        f"space_charge_limit_yield {HOBBS_WESSON_CRITICAL_YIELD_XE:.4f} = 1 - 8.3 sqrt(m_e / M_Xe) is a diagnostic threshold only). Surface charge changes by absorbed "
        "MINUS emitted; ke_see_emitted_j is an injected ledger term. Every field of this block enters config_sha256 through SEEConfig.to_dict; the SEE draws use "
        "their own RNG stream (CPU stream 5, Warp seed-table column 4), so the MCC / injection / ion-MCC draws of the reference are reproduced step for step "
        "until the first emission. The R2 box shakedown of the ss-v4 protocol with this block (4ca89e72, 2026-09-04 17:26 UTC, non-evidentiary, 100k steps) "
        "read I_d 2.06x the SEE-off shakedown's transient value and per-cusp effective yields 0.96-1.00 with NEGATIVE near-wall drops (a virtual cathode) at all "
        "three cusps: the cusps sat at the Hobbs-Wesson limit - the per-cusp SCL flag is therefore a primary reading of this campaign, not an afterthought")
    return block


def _collision_set_block() -> dict[str, Any]:
    block = copy.deepcopy(COLLISION_SET_V2_BLOCK)
    block["collision_set_note"] = (
        f"model v2.3.0 {XE_COLLISION_SET_V2_NAME} (spec pic2d-model-v2.3.json): electron set {XE_ELECTRON_SET_V2_FILE} (payload sha256 "
        f"{XE_ELECTRON_SET_V2_PAYLOAD_SHA256[:12]}; elastic + ionisation byte-identical to the legacy v1 set, the lumped 8.32 eV excitation split into the "
        "Biagi-v7.1 levels 8.315 / 9.447 / 9.917 / 11.7 eV whose sum equals the lumped table to 0.24 % above 10 eV) + ion-neutral set "
        f"{XE_ION_NEUTRAL_SET_V1_FILE} (payload sha256 {XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256[:12]}; CEX = Miller 2002 (87.3 - 13.6 log10 E) A^2, MEX = Phelps "
        "isotropic 3.39e-19 E^-1/2 m2, E = 1/2 M |v_i - v_n|^2, null-collision operator per ion sub-step against the inventory density with a Maxwellian atom "
        "at neutral_temperature_k; CEX fast neutrals: straight-line march through the cell mask -> exit (inventory sink F, pz_fast_neutral_exit) / wall "
        "(thermalises, pz_fast_neutral_wall) / thermal). ion_neutral: true selects both processes with the default table grid (0.05 eV steps to 2000 eV, "
        "fast-neutral speed threshold 4 v_th). The hashes are NOT read from the protocol: the named set is loaded from the spec files and its recomputed "
        "payload hashes enter config_sha256 (MCCConfig.collision_set). The ion MCC draws use their own RNG stream (Warp seed-table column 3), so the "
        "electron MCC / injection draws of the reference are reproduced step for step. The R3 box shakedown of the ss-v4 protocol with this block "
        "(6defd5ed, 2026-09-04 17:04-17:14 UTC, non-evidentiary, 100k steps) read CEX 9.4e14 /s and MEX 4.9e14 /s against S 1.6e16 /s (CEX / S 5.8 %), "
        "0.31 CEX events per exit ion, fast neutrals 324 exit / 3103 wall / 674 thermal, level shares 22 / 20 / 40 / 18 %, residual +0.09 % with the "
        "ion_neutral_loss_j sink booked")
    return block


def compose_case_protocol(case_id: str, *, wall_budget_seconds: float | None = None, budget_note: str | None = None) -> dict[str, Any]:
    """The ss-v4 protocol with the physics-effects changes for ``case_id`` (deterministic; sealed under ``protocols/``)."""

    if case_id not in CASES:
        raise KeyError(f"unknown case {case_id!r}; cases: {sorted(CASES)}")
    case = CASES[case_id]
    p = copy.deepcopy(load_v4_protocol())
    p["schema_version"] = CASE_SCHEMA_VERSION
    p["experiment_id"] = f"{EXPERIMENT_ID}-{case_id}"
    changes = ["numerics.peak_debye_gate.min_accumulated_macro_particle_steps_at_peak (v2.0.6 floor)", "numerics.performance.moment_sample_interval = 5 (v2.0.5)",
               "stopping_rule.wall_budget_seconds (launch-box measured rate x 1.5)", ("stopping_rule.acceptance (b corrected ledger, c shift table + per-cusp / effect "
               "diagnostics, hypotheses, d verdict + additivity)"), "case.id", "status / classification / model_version / model_spec / claim_boundary / simplifications text",
               "reference_run -> the ss-v4 plateau as the SEE-off / legacy-set point with its corrected-ledger status"]
    if case["see"]:
        changes.insert(0, "numerics.see (model v2.2.0 see_dielectric_v1, BN preset)")
    if case["collision_set"]:
        changes.insert(0, "operating_point.collision_set (model v2.3.0 xe_collision_set_v2, ion_neutral true) + the documentary cross_sections entry")
    p["campaign"] = {"experiment_id": EXPERIMENT_ID, "case": case_id, "label": case["label"], "effects": list(case["effects"]), "see": bool(case["see"]),
                     "collision_set": bool(case["collision_set"]), "role": case["role"], "launch_priority": list(LAUNCH_PRIORITY),
                     "template": "modern/experiments/pic2d_cft_steady_state_v4/protocol.json (the SEE-off / legacy-set / alpha = 0 run; every key not named in "
                                 "campaign.changes is byte-for-byte its value)",
                     "changes": changes,
                     "anomalous_transport": "NONE (alpha = 0): the R1 alpha-series (pic2d_anomalous_transport_v1) carries the transport closure; here each effect is "
                                            "isolated against the recorded ss-v4 plateau, so the physics identity of a case differs from v4's by the effect block(s), "
                                            "K = 5 and the declared Debye floor only"}
    p["status"] = "preregistered_physics_effects_see_collision_set_not_validated"
    p["classification"] = ("axisymmetric_electrostatic_pic_mcc_physics_effects_" + "_".join(e for e in case["effects"]) +
                           "_on_the_33um_reference_plateau_v1_3_closure_v2_0_6_gates_not_validated")
    p["model_version"] = MODEL_VERSION
    p["model_spec"] = MODEL_SPEC
    num = p["numerics"]
    if case["see"]:
        num["see"] = _see_block()
    if case["collision_set"]:
        p["operating_point"]["collision_set"] = _collision_set_block()
        p["cross_sections"] = {"electron": f"modern/spec/pic2d/{XE_ELECTRON_SET_V2_FILE} (payload sha256 {XE_ELECTRON_SET_V2_PAYLOAD_SHA256})",
                               "ion_neutral": f"modern/spec/pic2d/{XE_ION_NEUTRAL_SET_V1_FILE} (payload sha256 {XE_ION_NEUTRAL_SET_V1_PAYLOAD_SHA256})",
                               "note": "documentary: the runner loads the electron set named by operating_point.collision_set and hash-checks it; the legacy "
                                       "v1 file (the reference's) is modern/spec/pic2d/xenon-cross-sections-v1.json"}
    gate = num["peak_debye_gate"]
    gate["min_accumulated_macro_particle_steps_at_peak"] = 64000
    gate["min_accumulated_macro_particle_steps_at_peak_note"] = (
        "v2.0.6 (spec gates_v2_0.peak_debye_gate_accumulated_floor_v2_0_6): the gated node is the densest node whose ACCUMULATED macro-electron-steps over the "
        "400 000-step window reach 64 000 (= 32 crossings x ~2000 steps), so axis columns that hold < 32 macro-electrons per step but are visited by many are "
        "gate-able (the v2.0.3 mean-occupancy floor of 32 made them invisible); on the v4 maps the gated statistic is unchanged (2.154 at the same node, "
        "resolved nodes 19 650 -> 42 130); the mean-occupancy floor stays recorded alongside")
    num["performance"] = {"moment_sample_interval": 5,
                          "moment_sample_interval_note": "v2.0.5: electron window moments (T_e maps, peak-Debye T_e, sample counts) sampled every 5th accumulated step; "
                                                         "physics bitwise, gated Delta/lambda_D moves by 1.7e-5 median vs K = 1 (8aca6c3a); enters config_sha256 by the "
                                                         "v2.0.5 identity policy (K != 1 is a declared configuration); the reference ran K = 1"}
    num["frame_recorder_note"] = ("v2.0 frame recorder ON (28 ns frames): the effect's ionisation / density / potential structure at the cusp planes against the v4 "
                                  "frames (0d228ad2) - with SEE the frames gain wall_see_flux_per_m2_s / wall_see_effective_yield / wall_see_mean_energy_ev; "
                                  "frames are diagnostics, not gates")
    p["case"]["id"] = f"{case_id}-33um-dt1.4ps-w2.667e4-ng0-5.5e19-seed5e16-inventory-v1.3-closure"
    p["case"]["seed_note"] = ("seed 20260903 = the ss-v4 seed: the new processes draw from their own RNG streams (ion MCC: Warp seed-table column 3; SEE: column 4 / "
                              "CPU stream 5), so the electron-MCC and injection draws of the reference are reproduced step for step until the first new event")
    p["budget_v1_3"]["cost_model"] = {
        "source": "the R1 alpha-series launch-box preflights (same 90 x 720 grid, W 26 666.7, ss-v4 template, K = 5, H100 as the 4th CUDA-MPS client, 16:48-16:52 UTC "
                  "2026-09-04): 4.77 ms/step at 2.26 M e- + 2.26 M i -> 6.8 h to 3 transits; the R2 shakedown (SEE on) ran 5.17 ms/step and the R3 shakedown "
                  "(collision set v2) 4.52 ms/step at the seed load as 5th clients (audit estimates: SEE +2-4 % per step, ion MCC per sub-step small)",
        "steps_to_3_transits": STEPS_TO_3_TRANSITS,
        "a_priori_hours_to_3_transits": "7-8 h in an MPS slot (4.8-5.5 ms/step)",
        "gpu_memory_estimate_gb": "~1.5-2 (the v4 run's device pool + the SEE birth reservation / ion-MCC tables; 4.5 M macro-particles at the plateau)",
        "budget_basis": "wall_budget_seconds = 1.5 x the launch-box preflight rate at the plateau load x steps_to_3_transits, recorded in preflight-<case>.json before the preregistration commit",
    }
    stop = p["stopping_rule"]
    if wall_budget_seconds is not None:
        stop["wall_budget_seconds"] = float(wall_budget_seconds)
        stop["wall_budget_note"] = budget_note or "1.5 x the launch-box measured plateau-load rate x 3-transit steps (preflight-<case>.json)"
    else:
        stop["wall_budget_seconds"] = 43200.0
        stop["wall_budget_note"] = ("A-PRIORI 12.0 h = 1.5 x 8.0 h (5.6 ms/step in an MPS slot x 5.14 M steps); REPLACED by the launch-box measured rate x 1.5 before the "
                                    "preregistration commit (compose --budget-from-preflight)")
    stop["fail_closed"] = stop["fail_closed"].replace("v2.0.3 window-mode peak-node Debye gate", "v2.0.6 window-mode peak-node Debye gate (accumulated-particle-step floor)") \
        .replace("v2.0.3 windowed residual-power bound", "v2.0.3 windowed residual-power bound on the v2.0.6 W-corrected ledger")
    if case["collision_set"]:
        stop["fail_closed"] += "; v2.3.0: an ion-MCC null-collision ceiling violation ends the run at the next series record (PIC2DStabilityError)"
    if case["see"]:
        stop["fail_closed"] += "; v2.2.0: an overflow of the Warp SEE birth reservation (256 + N_e / 1000 per step) fails closed at the sync"
    stop["grid_heating_triad"]["note"] += ("; v2.0.6: the ledger's inelastic_loss_j carries W, so the windowed statistic IS the corrected one (the recorded ss-v4 "
                                           "series read -7.7 % where the corrected value was +2.46 %); the new sinks / sources (ion_neutral_loss_j, ke_see_emitted_j) "
                                           "are booked, so the residual stays the numerical-heating witness")
    hypotheses = HYPOTHESES_BY_CASE[case_id]
    stop["acceptance"] = {
        "declared": "predeclared before the launch; evaluated by `run.py assess --case <case>` (per case) and `run.py assess --campaign` (the three cases and the "
                    "additivity statement) against reference_run (the recorded ss-v4 plateau); verdicts recorded in results/<case>/assessment.json and "
                    "results/campaign-assessment.json (results-only commits)",
        "a_plateau": "stop_reason == plateau_reached_after_min_transit_times under the v4 rule (>= 3 transits = 5 142 858 steps, trailing-20 % drifts of I_d, N_e, n_g "
                     "< 5 %, triad soft bounds, window-mode peak-Debye soft margin 2.5)",
        "b_residual_power": "summary.grid_heating_triad.windowed_energy_residual_over_electrode_work (trailing 400 000-step ratio at the stop, v2.0.6 W-corrected "
                            "ledger) < +0.02, one-sided; the reference reads +0.0246 on the corrected ledger (FAIL) - a case that passes (b) is a cleaner plateau "
                            "than the reference",
        "c_shifts": {
            "quantities": list(QUANTITY_KEYS),
            "rule": "for every quantity the shift (case - reference) is reported: relative for the plateau quantities with the ss-v4 particle band, ABSOLUTE for the "
                    "IEDF low-energy fraction with the declared band 0.03. A shift with a declared '+' / '-' hypothesis counts as CONFIRMING when it has the declared "
                    "sign AND exceeds the band, CONTRADICTING when it has the opposite sign AND exceeds the band, INSIDE THE BAND otherwise; a '0' hypothesis is "
                    "CONFIRMING inside the band and CONTRADICTING beyond it in either direction; quantities without a band (anode_ion_a, wall_electron_power_w, "
                    "wall_ion_mean_energy_ev, the collision / fast-neutral rates) are REPORTED with their hypothesis sign and never judged",
            "hypotheses": hypotheses,
            "key_quantities": list(KEY_QUANTITIES[case_id]),
            "particle_band": PARTICLE_BAND,
            "absolute_band": ABSOLUTE_BAND,
            "per_cusp": {"planes_m": list(CUSP_PLANES_M), "half_width_m": CUSP_HALF_WIDTH_M,
                         "report": "per cusp plane (+-1 mm): electron and ion wall current (maps.npz wall fluxes x wall area), the axis-to-wall potential drop, the "
                                   "near-wall drop phi[wall - 3 cells] - phi[wall] (negative = non-monotonic sheath / virtual cathode), the near-wall T_e, the "
                                   "wall-ion mean impact energy; with SEE: the effective yield (emitted / impacting over the window, from wall_see_flux_per_m2_s), the "
                                   "SEE current, the mean emitted energy and the SCL flag (effective yield >= space_charge_limit_yield 0.983 OR near-wall drop < 0); "
                                   "reported beside the v4 values, signs judged by the hypothesis rows cusp_sheath_drop_v (never in the verdict: no band is measured)"},
            "effect_diagnostics": {
                "see": ["window_currents_a.see_emission_a / see_effective_yield", ("series see_* trailing-20 % means (wall potential mean / min / max, plasma-minus-wall "
                        "drop, cumulative effective yield, emitted power, backscattered fraction)"), "cusps at or above the Hobbs-Wesson limit (count of 3)"],
                "collision_set": ["window_currents_a.cex_rate_per_s / mex_rate_per_s / fast_neutral_exit_rate_per_s / fast_neutral_wall_rate_per_s", "CEX / S",
                                  "exit-plane IEDF: low-energy fraction (< 30 eV), fraction below 0.5 x anode, mean / peak energy vs the v4 IEDF",
                                  ("fast-neutral exit momentum rate (pz_fast_neutral_exit over the trailing window from series.jsonl; run average as the witness) and "
                                   "exit kinetic-energy rate"), "level shares of the four excitations"],
            },
        },
        "d_verdict": {
            "plateau_status": {
                "plateau_clean": "(a) AND (b): a quotable plateau of the effect",
                "plateau_heating": "(a) but NOT (b): the plateau heats above 2 % (like the reference); the shifts are reported with the heating caveat",
                "no_plateau": "NOT (a): budget / gate stop; trailing-window quantities reported",
            },
            "per_case_hypothesis_verdict": {
                "confirmed": "(a) reached AND every key quantity is CONFIRMING AND no hypothesis quantity with a band is CONTRADICTING",
                "not_confirmed": "(a) reached AND at least one hypothesis quantity with a band is CONTRADICTING - a finding about the effect's sign, recorded as such",
                "inconclusive": "NOT (a), OR (a) reached with no contradiction but some key quantity INSIDE THE BAND (the effect is smaller than the particle band)",
            },
            "combined_vs_sum_of_parts": {
                "applies_to": "see-bn+xe-set-v2, evaluated by `assess --campaign` once all three cases have an assessment",
                "rule": "for every banded quantity: interaction = shift(combined) - [shift(see-bn) + shift(xe-set-v2)]; ADDITIVE when |interaction| <= band, "
                        "SUPER_ADDITIVE when |interaction| > band and interaction has the sign of the sum of parts (the combined effect exceeds the sum), "
                        "SUB_ADDITIVE when |interaction| > band and it opposes the sum (the effects partly cancel / saturate); the statement is 'additive' when every "
                        "banded quantity is ADDITIVE, 'interacting' otherwise, 'not_evaluable' unless all three cases reached (a)",
                "hypothesis": "no interaction sign is predeclared: SEE acts through the sheaths, the collision set through the ion population; the audit gives no "
                              "reason to expect either sign, so the statement is a measurement",
            },
            "note": "the verdict is about the SIGN of each effect on the recorded plateau; the magnitudes are the measurement. Neither effect is 'tuned': the BN "
                    "constants and the cross sections are the declared literature values",
        },
    }
    dropped = ("development/screening run", "single seed and a single refined grid", "preregistered resolution-convergence")
    if case["collision_set"]:
        dropped += ("no ion-neutral collisions", "electron-neutral: isotropic elastic")
    if case["see"]:
        dropped += ("dielectric wall: perfectly absorbing",)
    p["simplifications"] = [s for s in p["simplifications"] if not s.startswith(dropped)]
    if case["collision_set"]:
        p["simplifications"] += [
            ("electron-neutral: isotropic elastic without the 2 m_e/M energy loss, FOUR Biagi-v7.1 excitation levels (8.315 / 9.447 / 9.917 / 11.7 eV; no metastable pool, no "
             "stepwise ionisation), single ionisation with Vahedi-Surendra secondary energy and isotropic emission (model v2.3.0)"),
            ("ion-neutral: Xe+ - Xe charge exchange (Miller 2002) and momentum transfer (Phelps isotropic) against the 0-D inventory density with a Maxwellian atom at the "
             "gas temperature; CEX fast neutrals are booked by a straight-line march, not tracked as particles; no Xe2+, no two-temperature atom population"),
        ]
    if case["see"]:
        p["simplifications"] += [
            ("dielectric wall: secondary electron emission from BN (Vaughan fit of Villemant 2019 + Sydorenko split, T_see 2 eV, ion-induced yield 0, no Hobbs-Wesson cap - the "
             "space-charge-limited state emerges) with accumulated surface charge = absorbed - emitted; zero field in the dielectric backing (perfect insulator); no sputtering, "
             "no surface conditioning / aging (Tondu 2011), no energy-dependent backscatter fractions"),
        ]
    p["simplifications"] += [
        "no anomalous cross-field transport (alpha = 0): the R1 alpha-series carries that closure; every discharge quantity here is conditional on alpha = 0 (audit section 6)",
        ("single seed per case: the shifts are judged against the recorded 50 um particle band, not a per-case replicate; the IEDF low-energy fraction has no replicate band "
         "(declared absolute band 0.03)"),
        "the reference (ss-v4) heats at +2.46 % of the electrode work on the corrected ledger: differences against it carry that caveat",
        "preregistered physics-effect study of a development model: no experimental validation, not a performance prediction",
    ]
    p["claim_boundary"] = ("preregistered physics-effects campaign (secondary electron emission from the BN wall, the xenon collision set v2 with Xe+ / Xe CEX and MEX, and "
                           "both together) on the reference design at 33.3 um / 1.4 ps / W 2.667e4 under the v1.3 closure, the v4 operating point and alpha = 0, against the "
                           "recorded ss-v4 plateau; the outcome is the SIGN of each effect on I_d, S, utilisation, n_g, I_beam, peak n_e, T_e,peak, the exit-plane IEDF, "
                           "the per-cusp sheath drops / effective yields / SCL state and the fast-neutral bookkeeping, with the magnitudes recorded, and whether the two "
                           "effects add; every discharge quantity of the 2D axisymmetric model is conditional on the declared transport closure (here none); not "
                           "validated against experiment; not a thruster performance prediction; the neutral transient is artificial and only the fixed point is physical")
    p["reference_run"] = v4_reference_block()
    p["preregistration"] = {
        "protocol": "protocols/<case>.json is frozen at the preregistration commit (its sha256 is listed in the campaign protocol.json); summary.json records protocol_sha256 "
                    "and git_head; run.py launch refuses a dirty worktree, a HEAD that is not the recorded preregistration commit, a sealed protocol that differs from its "
                    "recomposition, or an existing execution lock",
        "preflight": "preflight-<case>.json on the launch box (real P2 field on the 90 x 720 grid, mesh, factorisation, memory, ms/step at the seed load and at a synthetic "
                     "~4.5 M-particle load, GPU load before) - non-evidentiary; the budget is derived from it",
        "shakedown": "shakedown-<case>.json: a 100 000-step real-input run of EVERY case (results-shakedown/<case>/, cadences shrunk, every gate live) through run -> "
                     "finalize -> assess (case and campaign) on the launch box - non-evidentiary, not committed beyond the record; the earlier R2 / R3 shakedowns "
                     "($WORK/r2/shakedown-see-bn.json at 4ca89e72, pic2d_xe_collision_set_v2_shakedown at 6defd5ed) ran other code trees and other protocols "
                     "(no K = 5, no Debye floor, pre-rebase seed-stream layout) and are NOT reused as this campaign's shakedowns",
        "one_execution": "one detached launch per case from the scheduler's worktree at the preregistration commit, one MPS slot each, in the declared priority order as "
                         "slots free (after the R1 queue: ext-val bohm-0.4 -> alpha-1over64 -> alpha-0.345); a wall-budget stop may be resumed (new session, same "
                         "identity, disclosed); no parameter is changed after the freeze",
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
        "status": "preregistered_physics_effects_see_collision_set_not_validated",
        "title": "Physics effects v1: secondary electron emission from the BN wall (R2) and the xenon collision set v2 (R3), alone and together, on the 33 um reference plateau",
        "model_version": MODEL_VERSION,
        "model_spec": MODEL_SPEC,
        "design": {
            "template": "modern/experiments/pic2d_cft_steady_state_v4/protocol.json (reference design divergent-exit-stack, 90 x 720 at 33.33 um, dt 1.4 ps, W 26 666.7, "
                        "v1.3 closure, seed 20260903, frames ON); alpha = 0 (no anomalous-transport block)",
            "cases": {k: {"label": v["label"], "effects": list(v["effects"]), "see": v["see"], "collision_set": v["collision_set"], "role": v["role"]} for k, v in CASES.items()},
            "reference_point": "the recorded ss-v4 plateau (0d228ad2): SEE off, legacy collision set, alpha = 0, K = 1 - NOT re-run",
            "launch_priority": list(LAUNCH_PRIORITY),
            "launch_priority_note": "see-bn first (the largest predeclared effect, and the cusp SCL question), then xe-set-v2, then the combined case (whose additivity "
                                    "statement needs both single-effect plateaus); each takes one H100 MPS slot AFTER the R1 queue (ext-val bohm-0.4 -> alpha-1over64 -> "
                                    "alpha-0.345) as the scheduler frees one; never a 5th client",
            "see_block": SEE_BN_BLOCK,
            "collision_set_block": COLLISION_SET_V2_BLOCK,
            "hypotheses_by_case": HYPOTHESES_BY_CASE,
            "key_quantities_by_case": KEY_QUANTITIES,
            "cusp_planes_m": list(CUSP_PLANES_M),
        },
        "acceptance": {case: p["stopping_rule"]["acceptance"] for case, p in case_protocols.items()},
        "reference_run": first["reference_run"],
        "sealed_protocols": {f"modern/experiments/pic2d_physics_effects_v1/protocols/{case}.json": protocol_sha256(p) for case, p in case_protocols.items()},
        "budgets": budgets or {case: {"wall_budget_seconds": p["stopping_rule"]["wall_budget_seconds"], "note": p["stopping_rule"]["wall_budget_note"]} for case, p in case_protocols.items()},
        "identity_policy": {
            "reference": "SEE absent, collision set absent, hook absent, K = 1 -> config_sha256 f10772b25b03... = the ss-v4 record (test-pinned): the campaign's reference IS the recorded run",
            "see": "numerics.see enters config_sha256 through SEEConfig.to_dict (model name, every field, the resolved BN constants with provenance, the emission contract)",
            "collision_set": "operating_point.collision_set enters config_sha256 through MCCConfig.to_dict()['collision_set'] (recomputed payload hashes of the spec files, the process "
                             "lists with thresholds, the ion-neutral grid and speed threshold)",
            "k_5": "moment_sample_interval 5 enters config_sha256 (v2.0.5 policy); physics bitwise vs K = 1, diagnostics differ at <= 1.6e-3 relative (8aca6c3a)",
            "debye_floor": "min_accumulated_macro_particle_steps_at_peak 64000 enters config_sha256 only because it is declared (v2.0.6 policy)",
            "ledger": "the v2.0.6 W correction is code (bug fix, identity unchanged); acceptance (b) reads the corrected statistic natively",
            "alpha": "no anomalous_collisions block: the transport closure is not part of this campaign's identities",
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


def see_config_of(case_id: str) -> SEEConfig | None:
    """The SEEConfig a case declares (None for the SEE-off cases) - what the runner builds from ``numerics.see``."""

    if not CASES[case_id]["see"]:
        return None
    return SEEConfig(**{k: v for k, v in SEE_BN_BLOCK.items()})


__all__ = [
    "ABSOLUTE_BAND", "CASES", "COLLISION_SET_V2_BLOCK", "CUSP_HALF_WIDTH_M", "CUSP_PLANES_M", "EXPERIMENT_ID", "HYPOTHESES_BY_CASE", "HYPOTHESES_COMBINED",
    "HYPOTHESES_SEE", "HYPOTHESES_XE", "IEDF_FRACTION_BAND", "IEDF_LOW_ENERGY_FRACTION_OF_ANODE", "KEY_QUANTITIES", "LAUNCH_PRIORITY", "PARTICLE_BAND",
    "QUANTITY_KEYS", "REFERENCE_CASE", "REFERENCE_CORRECTED_RESIDUAL", "SEE_BN_BLOCK", "STEPS_TO_3_TRANSITS", "channel_wall_cells", "compose_campaign",
    "compose_case_protocol", "iedf_low_energy_fraction", "load_campaign", "load_case_protocol", "load_v4_protocol", "protocol_sha256", "reference_extras_from_v4",
    "see_config_of", "v4_reference_block", "wall_area_m2", "wall_power_and_ion_energy", "write_sealed_protocols",
]
